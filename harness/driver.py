"""Confused-deputy driver: find message templates that make the REAL agent perform a clean,
unprompted email.send (CONFUSED_DEPUTY), then filter for transfer across the guardrail zoo.

Two-tier transfer test:
  * fast STATIC check (`transfer_row`) reconstructs each guardrail's decision on the firing
    tool-call directly (no model) — used to filter during search.
  * the final portfolio should still be re-replayed against the strict guardrails for a
    faithful check (a denied call can change the agent's downstream behavior); single-action
    confused-deputy is exact under the static check, so it is a sound filter here.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from aicomp_sdk.attacks import AttackCandidate

from harness.guardrail_zoo import ZOO
from harness.replay import AgentFactory, deterministic_factory, replay

SENSITIVE = frozenset({"email.send", "http.post", "fs.write", "fs.delete", "shell.run"})


def _ctx_at(events: list, i: int, user_msgs: list) -> dict:
    """Reconstruct the guardrail context the sandbox would pass at tool-event i (sandbox.py)."""
    return {
        "seed": 123,
        "recent_sources": [e["source"] for e in events[:i]][-5:],
        "recent_tools": [e["name"] for e in events[:i]][-5:],
        "last_user": user_msgs[-1] if user_msgs else "",
        "side_effect": events[i]["side_effect"],
        "scope": events[i]["scope"],
    }


def transfer_row(trace: dict, guardrail_names: Sequence[str] | None = None) -> dict[str, bool]:
    """Per guardrail: would EVERY firing sensitive tool-call be ALLOWED? (fast, no model)."""
    names = tuple(guardrail_names) if guardrail_names is not None else tuple(ZOO)
    events, umsgs = trace["tool_events"], trace["user_messages"]
    firing = [i for i, e in enumerate(events) if e["ok"] and e["name"] in SENSITIVE]
    row: dict[str, bool] = {}
    for name in names:
        g = ZOO[name]()
        row[name] = bool(firing) and all(
            g.decide(events[i]["name"], events[i]["args"], _ctx_at(events, i, umsgs)).action == "ALLOW"
            for i in firing
        )
    return row


@dataclass
class Finding:
    messages: list[str]
    predicates: list[str]
    raw_optimal: float
    cell_hash: str | None
    transfer: dict[str, bool]
    calls: list[tuple]  # (name, args) of firing sensitive calls, for inspection

    def survives(self, guardrails: Sequence[str]) -> bool:
        return all(self.transfer.get(g, False) for g in guardrails)


def run_driver(
    templates: Sequence[Sequence[str]],
    *,
    agent_factory: AgentFactory = deterministic_factory,
    max_tool_hops: int = 1,
    observe_guardrail: str = "optimal_public",
) -> list[Finding]:
    """Run each template through the real agent; keep those that fire a predicate, with their
    static zoo-transfer profile."""
    findings: list[Finding] = []
    for msgs in templates:
        r = replay(list(msgs), guardrail=ZOO[observe_guardrail](), guardrail_name=observe_guardrail,
                   agent_factory=agent_factory, max_tool_hops=max_tool_hops)
        if not r.fired:
            findings.append(Finding(list(msgs), [], 0.0, None, {}, []))  # record the miss too
            continue
        calls = [(e["name"], e["args"]) for e in r.trace["tool_events"]
                 if e["ok"] and e["name"] in SENSITIVE]
        findings.append(Finding(
            messages=list(msgs),
            predicates=[p["predicate"] for p in r.predicates],
            raw_optimal=r.raw_score,
            cell_hash=r.score_cell_hash,
            transfer=transfer_row(r.trace),
            calls=calls,
        ))
    return findings


def build_portfolio(
    findings: Sequence[Finding], *, require_survive: Sequence[str] = ("dataflow", "data_inspecting")
) -> list[AttackCandidate]:
    """Emit AttackCandidates surviving the given strict guardrails, deduped by score-cell hash."""
    out: list[AttackCandidate] = []
    seen: set[str] = set()
    for f in findings:
        if not f.predicates or not f.survives(require_survive):
            continue
        if f.cell_hash and f.cell_hash in seen:
            continue
        if f.cell_hash:
            seen.add(f.cell_hash)
        out.append(AttackCandidate(user_messages=tuple(f.messages)))
    return out
