"""Candidate-sizing model for the JED attack replay (CLAUDE1.md §11.8).

The evaluator replays candidates IN ORDER until a per-phase deadline (9000 s). attack.py cannot
tailor per model (AttackRunConfig has no model id), so ONE candidate list is replayed on both
gpt_oss (slower) and gemma (faster). The binding question is how many candidates fit.

Two regimes:
  * partial_credit=False (installed-SDK behavior): a replay overrun FORFEITS the whole run -> bind
    N to the SLOWER model's safe limit (gpt_oss). gemma finishes early / under-loaded but safe.
  * partial_credit=True (hosted-gateway behavior per staff 733058, not yet battle-tested): an overrun
    TRUNCATES -> push N to the FASTER model's limit (or the 2000 cap); the slower model just scores
    its prefix. This is a large gain, especially for gemma. CONFIRM with a deliberate-overshoot probe
    before relying on it.

Feed in per-candidate replay seconds measured on the real models (probe / GPU). Placeholder values
below are forum estimates ONLY (~25 s gpt_oss, ~12 s gemma) — do not treat as measured.
"""
from __future__ import annotations

PHASE_BUDGET_S: float = 9000.0     # host FAQ: per-phase replay budget
HARD_CAP: int = 2000               # MAX_REPLAY_FINDINGS
WARMUP_S: float = 200.0            # cold start (~90-110 s) + Pilkwang's ~146 s untimed warm-up cushion
SAFETY_MARGIN: float = 0.85        # leave headroom against timing variance when overrun is fatal


def safe_n_single_model(t_candidate_s: float, *, partial_credit: bool = False,
                        budget_s: float = PHASE_BUDGET_S, warmup_s: float = WARMUP_S) -> int:
    """Max candidates one model can replay within the phase budget."""
    if t_candidate_s <= 0:
        return HARD_CAP
    usable = max(0.0, budget_s - warmup_s)
    if partial_credit:
        n = int(usable / t_candidate_s)            # overrun truncates -> can fill to the line
    else:
        n = int(usable * SAFETY_MARGIN / t_candidate_s)  # overrun forfeits -> keep a margin
    return max(0, min(HARD_CAP, n))


def recommend_n(t_gpt_s: float, t_gemma_s: float, *, partial_credit: bool = False) -> dict:
    """Recommended shared candidate count + what each model actually replays."""
    n_gpt = safe_n_single_model(t_gpt_s, partial_credit=partial_credit)
    n_gem = safe_n_single_model(t_gemma_s, partial_credit=partial_credit)
    if partial_credit:
        # push to the faster model's limit (or cap); the slower model truncates safely to its prefix
        n = min(HARD_CAP, max(n_gpt, n_gem))
    else:
        # bind to the slower model so nothing overruns (gemma then under-loads but is safe)
        n = min(n_gpt, n_gem)
    return {
        "partial_credit": partial_credit,
        "recommend_n": n,
        "gpt_oss_replays": min(n, n_gpt if not partial_credit else safe_n_single_model(t_gpt_s, partial_credit=True)),
        "gemma_replays": min(n, n_gem if not partial_credit else safe_n_single_model(t_gemma_s, partial_credit=True)),
        "gpt_oss_limit": n_gpt, "gemma_limit": n_gem,
    }


if __name__ == "__main__":
    print("SIZING (PLACEHOLDER forum times ~25s gpt_oss / ~12s gemma — REPLACE with probe measurement):")
    for pc in (False, True):
        r = recommend_n(25.0, 12.0, partial_credit=pc)
        print(f"  partial_credit={pc!s:5} -> emit N={r['recommend_n']:4}  "
              f"(gpt_oss replays ~{r['gpt_oss_replays']}, gemma ~{r['gemma_replays']}; "
              f"limits gpt={r['gpt_oss_limit']} gem={r['gemma_limit']})")
    print("\nRead: without partial-credit, bind to gpt_oss (~300) and gemma under-loads. With partial-credit"
          " confirmed, push to ~600+ so gemma fills up and gpt_oss safely truncates. Probe measures the real t.")
