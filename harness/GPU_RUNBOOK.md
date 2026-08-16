# GPU SESSION RUNBOOK — one pass, two fronts

The GPU session answers the two questions that gate BOTH of our arms. It is research only — the
submission is still `attack.py`, scored on Kaggle. Target: ~2–3h, ~$3–20.

## What it decides (why it's worth doing)
1. **COVERAGE front — does gemma fire our coverage primitives (C/D/U/X)?** This is now the #1
   unknown: "MEAN + both fire" and "SUM + gemma≈0" fit our board data identically, so we have NO
   direct evidence gemma elicits coverage. If it doesn't, our collapse-world arm is HALF strength.
2. **THROUGHPUT front — is the gemma leg boostable?** Sub2=78 = boosted-gpt + unboosted-gemma. If
   gemma's per-post cost is CoT/verbosity-limited, a terse formulation lifts it → 78→~100+. If it's
   raw-generation-limited, the ceiling is real. This is the concrete lever past the ~78 wall.

## Box to rent
- **1× 24GB GPU** (RTX 4090 / A10) is enough: gpt-oss-20b is MXFP4 ~16GB, gemma-4-26B-A4B QAT-q4 ~9GB
  (MoE, ~4B active). Vast.ai / RunPod / Lambda. A single A100-80GB (unquantized) is the comfortable
  option if cost is no object — one short session.
- Linux; ~40GB disk for the two GGUFs + HF cache.

## Setup (on the box)
```bash
python -m venv .venv && . .venv/bin/activate
pip install llama-cpp-python transformers torch huggingface_hub   # gguf path needs llama-cpp-python
# bring the SDK bundle (the 12MB competition Data → Download All): aicomp_sdk + kaggle_evaluation
#   put it on PYTHONPATH exactly like the Mac (.pth or PYTHONPATH=/path/to/sdk_bundle)
# clone the repo for the harness:
git clone <repo> && cd <repo>/harness
# EDIT the hardcoded path in gpu_elicit.py if present; models auto-download from HF on first run:
#   gpt_oss: unsloth/gpt-oss-20b-GGUF ; gemma: unsloth/gemma-4-26B-A4B-it-GGUF
```
Verify the SDK imports: `python -c "import aicomp_sdk; from aicomp_sdk.evaluation.ops import resolve_fixtures_dir; print(resolve_fixtures_dir())"`

## Front 1 — coverage elicitation (run FIRST; it's the gating bit)
```bash
# allow_all = permissive: measures pure ELICITATION (does the model produce the trace at all)
python -m harness.gpu_elicit --agent gguf --model gpt_oss --guard allow_all --max-hops 8 --max-new-tokens 256
python -m harness.gpu_elicit --agent gguf --model gemma  --guard allow_all --max-hops 8 --max-new-tokens 256
```
**Read the per-primitive fire-rate table.** The decision it drives:

| gemma result | meaning | action |
|---|---|---|
| C fires, D fires | gemma has a real coverage leg (1-call prims) | keep C/D weighted; coverage arm is full-strength |
| C fires, D fires, **U/X 2-msg chains fire** | the gemma 1-call cap does NOT bind our chains | U/X extend to gemma → closes the content_plus_intent hole; can weight U up |
| U/X do NOT fire on gemma | 1-call cap binds multi-call prims | U/X are gpt_oss-only; shift their weight → D/C for the gemma leg |
| **C/D also fail on gemma** | gemma barely elicits coverage | coverage arm is ~half strength (gpt-only) → lower the top-5-via-coverage estimate; lean throughput |

Also check the X real-lineage line (`read_secret_ok` / `posted_after_read`): does the model post the
**actual** secret contents after a successful read? If no → X is dead even on a permissive guard; drop it.

Then run the same two commands with `--guard optimal_public` to confirm which primitives the PUBLIC
guard blocks (D/U should read as blocked — that's expected, NOT an elicitation failure).

## Front 2 — is the gemma throughput leg boostable?
```bash
python -m harness.gemma_throughput --agent gguf --reps 4
```
**Read the raw/sec table + the VERDICT line:**
- `gemma IS boostable` (best >1.3x plain) → adopt that formulation as Sub2's gemma route, rebuild
  Sub2, and re-submit — expect the board to climb from 78. This is the realistic path toward the pack.
- `~FLAT / generation-speed-limited` → no easy gemma boost; the ~78 ceiling is real; stop spending
  Kaggle slots on throughput and put the effort into coverage.
- `nothing fired` → a gemma parse failure, not a throughput result; fix the tool-call parser first.

## Parity caveats (do not skip)
- **Use `--agent gguf`, not `transformers`.** The hosted board runs GGUF-via-llama.cpp with the exact
  `KaggleGemma4ToolCallParser`; transformers uses a different parser and will give OPTIMISTIC gemma
  numbers (may not reproduce the 1-call cap). transformers is a research proxy only.
- Greedy decoding (`do_sample=False`) → deterministic → 1 rep per template is correct for fire-rate;
  the throughput probe still does a few reps to average wall-clock latency.
- **Local throughput ≠ hosted throughput** (a rented A100 ≫ Kaggle T4×2). The gemma probe tells you the
  REGIME (boostable vs flat) and the relative ranking of formulations — NOT the absolute hosted N.
  Confirm any winning formulation with a real Kaggle submission.
- The GGUF `chat_template` revision may differ from the scoring env; if gemma fire-rates look wrong,
  check the revision against `gemma_model_server.py`.

## What it CANNOT do (stay honest)
- It cannot reveal the private guard (held out, un-probeable) — that coin stays hidden.
- It cannot prove a formulation transfers to the hosted stack — Kaggle submission is the only ground
  truth. GPU narrows the search; the ~11h Kaggle slots confirm the winners.

## One-line decision summary after the session
- gemma fires coverage + boostable throughput → push BOTH arms hard, we're competitive in more worlds.
- gemma fires coverage, throughput flat → coverage arm is real; throughput capped at ~78; hedge holds.
- gemma barely fires coverage → coverage is gpt-only/half-strength → lean throughput + reconsider the
  two-final allocation.
