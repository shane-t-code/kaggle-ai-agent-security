# AI Agent Security — Multi-Step Tool Attacks

Automated red-teaming for tool-using LLM agents, built for the Kaggle **"AI Agent Security —
Multi-Step Tool Attacks"** competition (OpenAI / Google / IEEE, JED framework).
[**Competition link**](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks) · final standing: **Bronze medal — 391 / 4,186 teams**.

The task is defensive security research in a sealed, offline sandbox: submit an algorithm that
discovers reproducible, multi-step failures in tool-using agents — prompt injection, data
exfiltration, destructive writes, and confused-deputy actions — and have them replay-validated
against a guardrail. No real systems are touched; all attack content comes from the competition's
own seeded fixtures.

## By the numbers

| | |
|---|---|
| Final standing | Bronze — 391 / 4,186 teams |
| Target models | `openai/gpt-oss-20b`, `google/gemma-4-26B-A4B-it` (GGUF, greedy decode) |
| Agent versions / variants | 34 |
| Evaluation tools (`harness/`) | 46 |
| Evaluation hardware | rented RunPod GPU instances (A40 / A6000, 24–48 GB) |
| Committed iterations | 34 |
| Scored replay budget | up to 2,000 candidates/model, ≤ 8 tool hops/message |
| Peak public throughput score | ~95 (public board; private board decides rank) |

## The problem

Each submission is a `run(env, config) -> list[AttackCandidate]`; every candidate is a sequence of
user messages. During scoring the messages are **replayed** — `env.interact(msg)` drives a target
agent that may call tools across up to 8 hops, and the resulting tool-call trace is scored by four
predicates (exfiltration, untrusted-content-to-action, destructive write, confused deputy), weighted
by severity, plus a novelty bonus for distinct interaction "cells." The decisive structure is a
**public/private split**: candidates are replayed once against a permissive public guardrail and once
against a **stricter, held-out private guardrail with swapped fixtures**. Final rank is the best
**private** score of two manually selected submissions — and the private board is hidden until close.

## Runtime constraints

- `run()` may probe the live environment but returns a *fixed* list — generation does no heavy model
  work; the single scored pass is the static replay.
- Deterministic greedy decode, a fixed 8-hop cap per message, ≤ 2,000 candidates scored in order, one
  tool call per generation. Throughput is therefore a function of candidate *structure*, not sampling.
- The scored agent runs server-side; the only board-faithful local proxy is the same GGUF build served
  through `llama.cpp`, which this repo's harness reproduces.

## Approach

The strategy is a **two-submission hedge across the one unobservable unknown** — whether the private
guardrail blocks on argument *content* or on data *provenance* — since each world rewards a different
attack shape and the hedge takes the max of the two.

| Component | What it does |
|---|---|
| Throughput arm (`attack_sub2_multidistinct.py`) | Multi-hop sentinel posts to distinct domains in one interact, with reasoning-channel suppression so continuation hops stay terse — maximizes scored events per unit of inference budget. |
| Coverage arm (`attack_sub1_robust.py`) | A diversified predicate mix (confused-deputy, destructive-write, untrusted-to-action) sized so it stays non-zero across the strict-guardrail worlds where the sentinel method collapses. |
| Probe-select-fill | `run()` times the live model, auto-routes the best template per model, and fills the replay budget with margin. |
| Reverse-engineered scorer | The predicate/cell/guardrail logic was read out of the SDK and the hosted gateway and reproduced locally, so every design choice is checked against the real scoring code rather than assumed. |

## Alternatives built and rejected on measurement

Most ideas were killed by a measurement, not a hunch:

- **URL-byte minimization** (shorter `http.post` URLs to cut decode tokens) looked like a ~1.2× win on
  a naive per-token metric, but a balloon-rate probe showed scheme-less URLs *trigger* reasoning-channel
  balloons on continuation hops, making real per-post cost *worse*. Dropped.
- **Same-call multipost** (repeat the identical post) washed out; **distinct-domain multipost** was the
  version the model actually chains reliably — the difference was only visible on the board, so the local
  ratio was treated as a candidate, never a conclusion.
- **Gemma two-post batching** gained locally but washed on the board (serving-stack parser cap), so the
  gemma leg stayed single-post.

## How changes were tested

1. Reproduce the hosted scorer on the same GGUF build via `llama.cpp`, on rented RunPod GPU
   instances (board-faithful).
2. Prefer hardware-invariant metrics (decode-tokens-per-scored-post, fire-rate) over wall-clock, which
   differs between local and board hardware.
3. Use deterministic decode so a candidate's scored events reproduce run to run.
4. Treat any local raw/sec gain as a *candidate*; the board A/B is the only settle (local-to-board
   transfer is not assumed).
5. Model the private world with a guardrail "zoo" (including the organizers' own published guardrail
   examples) so a candidate must survive a *family* of stricter guards, not just the public one.

- *Killed example:* the URL-minimization lever above — a hardware-invariant balloon probe overturned a
  tempting but confounded local result.
- *Shipped example:* distinct-domain multi-hop suppression — reproduced across probes and confirmed on
  the board, it roughly doubled the throughput arm's score over the single-post baseline.

## Tools (`harness/`)

- `elicit_search.py` — UCB1 → evolutionary search over a template grammar, scored by real raw/sec.
- `token_cost_probe.py` — decode-tokens-per-post (hardware-invariant); `--balloon` reports balloon rate.
- `cpu_throughput_probe.py` — ranks continuation-suppression phrasings (CPU exposes continuation cost).
- `coverage_leg_probe.py` / `sub1_leg_probe.py` — per-primitive fire-rate for the coverage arm.
- `guardrail_zoo.py` + `organizer_guards/` — modeled private-guard variants used as a transfer test.
- `replay.py` — replays any candidate through the real sandbox and scores it with the SDK.
- `GPU_RUNBOOK.md` — how to stand the harness up on a GPU pod.

## Repo layout

```
attack_sub2_multidistinct.py   throughput arm (final)
attack_sub1_robust.py          coverage arm (final)
attack_sub*.py                 agent variants explored along the way
harness/                       evaluation harness + probes
harness/organizer_guards/      organizers' published guardrail examples (see License)
```

## Running

The agent is self-contained (imports only the competition SDK) — rename a chosen variant to
`attack.py` to submit. The evaluation harness runs on a RunPod GPU instance per `harness/GPU_RUNBOOK.md`:

```
python -m harness.elicit_search --agent gguf --model gpt_oss --guard allow_all --reps 3
python -m harness.token_cost_probe --model gpt_oss --balloon --reps 16
```

## License

This project's own code is released under the MIT License (see `LICENSE`).

`harness/organizer_guards/guardrail_pattern.py` and `guardrail_taint_tracking.py` are verbatim copies
of the competition organizers' published guardrail examples (mbhatt/competitionscratch), retained with
their original attribution and used only as local stress-test proxies.
