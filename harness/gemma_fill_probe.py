"""
GEMMA FILL-EFFICIENCY PROBE (2026-08-28) - is the gemma leg UNDER-FILLED or at its decode ceiling?

Web-Claude challenge: our "gemma ~72 is the shared ceiling everyone hits" is an INFERENCE about the
pack's hidden split, not a measurement. This converts it to a fact for ZERO board cost: run the REAL
attack_sub2_multidistinct.run() against a board-faithful gemma env and measure how much of the replay
budget its fill loop actually consumes, extrapolated to the 9000s board budget.

  board fill ratio ~ 0.98-0.99  => gemma fills replay optimally => 72 is a decode/prefill CEILING
                                    (under-fill hypothesis DEAD; nothing to fix).
  board fill ratio << 0.95      => the fill loop leaves replay budget on the table (probe/margin eats
                                    too much or an early-stop) => REAL under-fill => fixable => +gemma leg.

Reads the [fill] stderr line printed by run() (selected template + returned N + cost/cap) as ground truth;
the printed board-extrapolated ratio removes the small-budget probe-overhead distortion.

USAGE (pod, board-faithful gguf):  python -m harness.gemma_fill_probe --budget-s 1800
                                   python -m harness.gemma_fill_probe --model gpt_oss --budget-s 1800  # sanity
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from harness.gpu_elicit import EnvRunner, build_factory
import attack_sub2_multidistinct as mp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma")
    ap.add_argument("--agent", default="gguf")
    ap.add_argument("--guard", default="allow_all")
    ap.add_argument("--budget-s", type=float, default=1800.0)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    factory = build_factory(args.agent, args.model, args.max_new_tokens)
    runner = EnvRunner(factory, args.guard, max_tool_hops=8)

    single = mp.TEMPLATES[0][1]  # 'plain' single-post - the template the selector routes gemma to
    lats = []
    for r in range(args.reps):
        t = time.monotonic()
        runner.run([mp._message(single, 990100 + r)])
        lats.append(time.monotonic() - t)
    lats.sort()
    c = lats[len(lats) // 2]

    cfg = SimpleNamespace(time_budget_s=args.budget_s, max_tool_hops=8)
    t0 = time.monotonic()
    cands = mp.AttackAlgorithm().run(runner.env, cfg)
    wall = time.monotonic() - t0
    n = len(cands)

    implied_replay = n * c
    overhead = max(0.0, wall - implied_replay)  # probe + non-firing + end-margin (~fixed vs budget)
    board_budget = mp.REPLAY_BUDGET_S
    board_cap = mp.REPLAY_SAFE * board_budget
    board_fill = max(0.0, board_budget - overhead) / board_cap if board_cap else 0.0

    print("\n### GEMMA FILL PROBE  model=%s guard=%s budget=%.0fs ###" % (args.model, args.guard, args.budget_s))
    print("  per-candidate cost c        = %.2fs  (median of %d single-posts)" % (c, args.reps))
    print("  run() emitted N             = %d candidates" % n)
    print("  run() wall                  = %.0fs   implied replay (N*c) = %.0fs" % (wall, implied_replay))
    print("  fixed overhead (probe+margin)= %.0fs" % overhead)
    print("  EXTRAPOLATED board fill @9000s = %.3f" % board_fill)
    print("  READ: ~0.98-0.99 => gemma fills replay optimally => 72 is a decode CEILING (under-fill DEAD).")
    print("        <<0.95      => real under-fill => fixable (emit sized-untrialed tail / trim probe) => +gemma leg.")
    print("  GROUND TRUTH: trust the [fill] line above (cost=X/Y) - if X~=Y at large budget, no under-fill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
