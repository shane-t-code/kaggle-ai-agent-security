"""
OVERHEAD PROBE — where does the board's per-candidate time go, and is it a REDUCIBLE lever?

WHY (2026-08-21): the faithful testbed (hosted_config_probe, n_gpu_layers=-1) proved the board = full-GPU:
multidistinct = 1.3x single (matches 78->90.63), toolonly = 1.0x (wash). So DECODE + PHRASING + generation-
amortization are all matched/exhausted. The only place the 90->114 gap can live is ABOVE one interact:
  (1) PER-CANDIDATE overhead: eval builds a FRESH SandboxEnv per candidate (ops.py:512-520) with the model
      reused warm. If that build+reset is a big fraction of a candidate's time, then packing MORE posts under
      ONE build (multi-MESSAGE: up to 32 msgs x 8 hops = 256 posts / 1 build) amortizes it = a real lever
      -- and multi-message's board WASH (Sabrina) would then be a handling bug, not a throughput verdict.
  (2) If per-candidate build is negligible, the board's extra cost is hosted RPC/scheduling (a separate
      model-server process, one round-trip per hop) which we CANNOT reproduce in-process -> board-A/B-only.

WHAT IT MEASURES (on the faithful full-GPU config, model built ONCE and reused warm):
  A. per-candidate cost breakdown: SandboxEnv BUILD vs reset() vs one 8-post interact.
  B. multi-message amortization: M messages under ONE env (1 build, M*8 posts) vs M separate candidates
     (M builds, 8 posts each) -> effective sec/post. If the 1-build version is meaningfully cheaper/post,
     multi-message IS a throughput lever on the faithful config (retest it on the board with correct structure).

USAGE (GPU pod, faithful config):
  python -m harness.overhead_probe --reps 5 --msgs 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
for _cand in (os.environ.get("AICOMP_SDK_BUNDLE", ""), str(_REPO / "sdk_bundle"),
              str(Path.home() / "kaggle_local" / "sdk_bundle")):
    if _cand and Path(_cand).exists() and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

from harness.hosted_config_probe import _hosted_gptoss_factory, _multidistinct, _count_posts, P

from aicomp_sdk.core.env.sandbox import SandboxEnv
from aicomp_sdk.evaluation.ops import resolve_fixtures_dir
from harness.guardrail_zoo import ZOO


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _build_env(agent, fixtures, hops):
    return SandboxEnv(seed=123, fixtures_dir=fixtures, agent=agent,
                      guardrail=ZOO["allow_all"](), max_tool_hops=hops)


def _count_env(env):
    tr = env.export_trace_dict()
    return sum(1 for e in tr.get("tool_events", [])
               if e.get("name") == "http.post" and e.get("ok")
               and P in str((e.get("args") or {}).get("data", "")))


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-candidate overhead + multi-message amortization on the faithful config.")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--msgs", type=int, default=4, help="messages per multi-message candidate")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-hops", type=int, default=8)
    args = ap.parse_args()

    print(f"\n### OVERHEAD PROBE: faithful full-GPU config, reps={args.reps} msgs={args.msgs} k={args.k} ###")
    print("    model built ONCE (warm) + reused; measures per-candidate SandboxEnv build/reset vs interact.\n")

    try:
        agent = _hosted_gptoss_factory(-1)()   # build the model ONCE, full GPU
        fixtures = resolve_fixtures_dir()
    except Exception as e:
        print(f"SKIP: could not build hosted gpt_oss backend: {e}", file=sys.stderr)
        return 2

    # warm-up
    try:
        w = _build_env(agent, fixtures, args.max_hops)
        w.reset(); w.interact(_multidistinct(990000, args.k), max_tool_hops=args.max_hops)
    except Exception as e:
        print(f"warm-up failed: {e}", file=sys.stderr)

    builds, resets, interacts, posts_each = [], [], [], []
    for r in range(args.reps):
        t0 = time.monotonic(); env = _build_env(agent, fixtures, args.max_hops); builds.append(time.monotonic() - t0)
        t0 = time.monotonic(); env.reset(); resets.append(time.monotonic() - t0)
        t0 = time.monotonic(); env.interact(_multidistinct(r, args.k), max_tool_hops=args.max_hops)
        interacts.append(time.monotonic() - t0); posts_each.append(_count_env(env))
        print(f"  cand{r}: build={builds[-1]*1000:6.1f}ms reset={resets[-1]*1000:6.1f}ms "
              f"interact={interacts[-1]:5.2f}s posts={posts_each[-1]}", flush=True)

    mb, mr, mi = _mean(builds), _mean(resets), _mean(interacts)
    mp = _mean(posts_each) or 1.0
    per_cand = mb + mr + mi
    overhead = mb + mr
    print(f"\n[breakdown] build={mb*1000:.1f}ms  reset={mr*1000:.1f}ms  interact={mi:.2f}s ({mp:.1f} posts)")
    print(f"[per-candidate] overhead(build+reset)={overhead*1000:.1f}ms  = {100*overhead/per_cand:.1f}% of the candidate")
    print(f"[per-post] single-candidate sec/post = {per_cand/mp:.3f}s (incl. overhead)")

    # --- B: multi-message amortization (M messages, ONE env build) ---
    print(f"\n[multi-msg] {args.msgs} messages under ONE env build:")
    try:
        t0 = time.monotonic(); env = _build_env(agent, fixtures, args.max_hops); mm_build = time.monotonic() - t0
        env.reset()
        mm_posts = 0; mm_interact = 0.0
        for j in range(args.msgs):
            t0 = time.monotonic(); env.interact(_multidistinct(5000 + j, args.k), max_tool_hops=args.max_hops)
            mm_interact += time.monotonic() - t0
        mm_posts = _count_env(env)
        mm_total = mm_build + mm_interact
        mm_per_post = mm_total / max(1, mm_posts)
        print(f"    build={mm_build*1000:.1f}ms  {args.msgs} interacts={mm_interact:.2f}s  posts={mm_posts}")
        print(f"    multi-message sec/post = {mm_per_post:.3f}s  (1 build amortized over {mm_posts} posts)")
        single_per_post = per_cand / mp
        gain = single_per_post / mm_per_post if mm_per_post else 0.0
        print(f"\n[key] single-candidate {single_per_post:.3f}s/post  vs  multi-message {mm_per_post:.3f}s/post "
              f"= {gain:.2f}x")
        if gain > 1.10:
            print(f"VERDICT: PER-CANDIDATE OVERHEAD IS A REAL LEVER — multi-message is {gain:.2f}x cheaper/post on the "
                  f"faithful config (amortizes the {overhead*1000:.0f}ms build+reset over {mm_posts} posts). Multi-message's "
                  f"board WASH (Sabrina) was likely a handling issue, NOT throughput -> RE-TEST multi-message on the board "
                  f"with correct single-candidate-many-messages structure. THIS could be the 90->114 lever.")
        else:
            print(f"VERDICT: per-candidate overhead is NEGLIGIBLE locally ({100*overhead/per_cand:.1f}%, multi-msg only "
                  f"{gain:.2f}x) -> the board's extra per-post cost is HOSTED (RPC round-trip per hop / scheduling), which "
                  f"we CANNOT reproduce in-process. => the 90->114 lever is board-A/B-only; local instruments have gone as "
                  f"far as they can. Test overhead-amortizing candidate structures directly on the board.")
    except Exception as e:
        print(f"    multi-message test failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
