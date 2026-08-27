# GPU SESSION RUNBOOK — next pod (turnkey)

## FRESH-POD SETUP (run EVERY time — stop/resume wipes disk + env). Copy-paste, top to bottom.
Pod: RunPod, template `Runpod Pytorch 2.8.0`, Jupyter ON, container disk 40GB, /workspace 60GB, any 24GB+ card
(A40 48GB $0.44/hr = value pick). In Jupyter, DRAG both zips into /workspace, then New -> Terminal:
```
cd /workspace
apt-get update -qq && apt-get install -y -qq unzip
unzip -o harness.zip
unzip -o ai-agent-security-multi-step-tool-attacks.zip -d sdk_bundle
ls sdk_bundle    # expect: aicomp_sdk  aicomp_sdk-3.1.2.dist-info  kaggle_evaluation  __init__.py
export AICOMP_SDK_BUNDLE=/workspace/sdk_bundle
export PYTHONPATH=/workspace/sdk_bundle:/workspace
export HF_HOME=/workspace/hf
pip install -q pydantic "gymnasium<1" openai huggingface_hub numpy
CMAKE_ARGS="-DGGML_CUDA=on" pip install -q llama-cpp-python --no-cache-dir   # CUDA build ~5-10 min
python -c "import llama_cpp; print('llama_cpp ok')"                          # verify build
```
Two zips to upload (both on Shane's Windows box, NOT git): `c:\Kaggle\harness.zip` (rebuild via the Python
`zipfile` forward-slash script; currently 48 files incl. gemma_fill_probe.py AND attack_sub2_multidistinct.py
at repo root — the fill probe imports it; if a probe errors ModuleNotFoundError on an attack_sub2_* file, drag
that file into /workspace) + `C:\Users\Shane\Downloads\
ai-agent-security-multi-step-tool-attacks.zip` (the Kaggle Data download = the SDK bundle). Re-export the 3 env
vars in EVERY new terminal. GGUFs auto-download to /workspace/hf on first run (gpt-oss ~12GB / gemma ~17GB;
"unauthenticated HF" warning is BENIGN, no token needed). Stop/resume may drop the HF cache + llama-cpp build
-> re-run the pip lines if `import llama_cpp` fails, and budget a re-download. See [[jed-runpod-gpu-workflow]].

## CURRENT PRIORITY (2026-08-29) — CONTINUATION-SUPPRESSION PHRASING RE-SEARCH (toolonly was board-VINDICATED)
```
python -m harness.cpu_throughput_probe --reps 4 --k 8 --threads 8
```
Board proved toolonly ~92.6 >> multidistinct ~73 (a 20pt PHRASING effect) -> the CPU wall-clock probe was RIGHT all
along and we wrongly demoted it as a "GPU artifact". This run broadens the continuation-suppression phrasing set (8
variants incl. REVIVED commentary_forge + toolonly_plus/noanalysis_hard/direct/numbered) and ranks them by CPU
raw/sec vs the toolonly INCUMBENT. CPU is board-faithful for continuation cost (GPU hides it). CPU 20B is SLOW
(~tens of s/interact) so this takes ~1-1.5h at reps=4. READ the VERDICT: it prints a BOARD SHORTLIST of any reliable
(8/8) challenger that beats toolonly by >5% -> board-test the top 2-3 with 3-4 draws EACH vs a fresh toolonly draw
(never 1-2 draws -- that small-sample trap is what mis-killed toolonly the first time). If none beats toolonly, the
phrasing space is near-maxed and the last ~14pts to medals is a board-serving/multipost-reliability effect, not a prompt.
NOTE: forces n_gpu_layers=0 (CPU) on purpose; the A6000 GPU sits idle -- that is correct, CPU is the faithful ranker.
(Old gemma_fill_probe result: gemma leg CONFIRMED at its decode ceiling ~72, no under-fill.)
The ONE pod task today. Everything else (the K-sweep) is BOARD-only — the pod fires every K cleanly so it can't
test board hop-degradation. This probe feeds the REAL attack_sub2_multidistinct.run() into a board-faithful gemma
env and measures how much of the replay budget its fill loop actually consumes, extrapolated to 9000s.
READ "EXTRAPOLATED board fill @9000s": ~0.98-0.99 => gemma fills replay optimally => the ~72 gemma leg is a real
decode/prefill CEILING (under-fill hypothesis DEAD, nothing to fix). <<0.95 => real under-fill => fixable => +~8-9
mean. Also trust the [fill] stderr line (cost=X/Y): X~=Y at large budget = no under-fill. gemma GGUF ~17GB downloads
first (~few min). ~30 min run at budget 1800. (Old throughput_prescreen/schednf/fewshot/statemach all board-DEAD.)

---


Board-faithful throughout: `--agent gguf` (llama.cpp on the hosted GGUFs; hosted parity = max_new_tokens 1024,
greedy, max-hops 8). `allow_all` guard measures ELICITATION/throughput (does the model perform the chain);
`optimal_public` measures board-faithful survival. Full pod setup (upload `harness.zip` + the Kaggle SDK bundle,
deps, env vars) is in memory **jed-runpod-gpu-workflow** — deploy any available 24GB+ card, one model at a time.
Interpreter on the pod is the venv python; LOCALLY the only interpreter with `aicomp_sdk` is
`C:\Users\Shane\anaconda3\python.exe`.

> GPU sessions run ~15 min, not 1h — do not under-budget. The whole plan below is ONE pass.

---

## PRIMARY INSTRUMENT — `harness/elicit_search.py` (the automated template SEARCH)
Built 2026-08-17 (workflow wf_aadf5e56). A simple two-phase search (UCB1 map → evolutionary mutate) over a
template GRAMMAR (framing × post-count × suppression-tail × arg-terseness × domain-scheme), reward = REAL
raw/sec through the real gguf model + real scorer, reusing one env via reset(). Greedy decode ⇒ raw is
deterministic (reps denoise latency only). Seeds + grammar in `harness/elicit_search_seeds.py` (37 verified
seeds; SDK-reproduced raw: multipost8=130, single=18, K=4→66). It supersedes the fixed `throughput_bandit.py`
(that stays as a fallback). It does NOT test board-transfer — it ranks by local raw/sec and FLAGS winners for
a board A/B. **Never treat a local raw/sec winner as a board gain — the board A/B is the only settle**
(multipost has a documented local→board non-transfer record: 78.75 neutral / 65.4 regression).

### What this session decides (from the synthesis)
1. **THE decisive unknown — the GEMMA cap.** Code-PROVEN it is NOT a Python wall (the sandbox banks up to 8
   successful hops; the only thing capping gemma at 1 is a model-EMISSION doubled-brace on continuation hops),
   so it is genuinely GPU-testable. STEP 3 settles it: if a gemma multi-hop arm banks `raw>18` (posts>1) the
   120 path has a mechanism; if EVERY gemma multipost arm returns `raw=18.0` exactly (posts=1.0) the cap is a
   real emission wall and **120 is out of reach by known means → stop throughput, commit to the coverage arm.**
2. **The gpt leg squeeze** — does distinct-domain multipost beat single-post raw/sec board-faithfully (78→~90-100).

### RUN PLAN (one ~1h pass; run top-down, stop early if STEP 3 collapses)
```
# STEP 1 — gpt_oss throughput (pure-elicitation guard)
python -m harness.elicit_search --agent gguf --model gpt_oss --guard allow_all --reps 3 --rounds 12 --budget-s 3600 --max-hops 8 --max-new-tokens 1024
#   GOOD: a K=8 multipost arm reports posts≈8.0 (raw≈130) at raw/sec > multi_distinct(61)×1.08 ≈ 66
#         → prints "FLAG WINNER … QUEUE FOR BOARD A/B".
#   BAD:  best arm is 'single' or raw/sec ≤ base×1.08 → "NO past-noise winner … ~88 is the earned ceiling".
#   Also read realized posts=(raw-2)/16 vs requested K — an arm that requests 8 but posts ~3 is auto-penalized.

# STEP 2 — gpt_oss board-faithful confirmation (only worth it if STEP 1 found a winner)
python -m harness.elicit_search --agent gguf --model gpt_oss --guard optimal_public --reps 3 --rounds 8 --budget-s 2400 --max-hops 8 --max-new-tokens 1024
#   GOOD: the winner survives the public guard at ~same raw/sec (clean-URL SECRET_MARKER ALLOWs+fires).
#   BAD:  raw/sec drops (an arm DENYs) → that arm is not board-real.

# STEP 3 — gemma: THE decisive test (grammar auto-drops Harmony tails on gemma; they error there)
python -m harness.elicit_search --agent gguf --model gemma --guard allow_all --reps 3 --rounds 8 --budget-s 3600 --max-hops 8 --max-new-tokens 1024
#   GOOD (cap BROKEN → opens 120): ANY multi-hop arm reports posts>1 (raw>18), ideally posts≈8/raw≈130.
#   BAD  (cap CONFIRMED → kills 120 by known means): EVERY multipost arm = raw≈18.0 exactly (posts=1.0).
#   THIS SINGLE NUMBER decides whether 120 is reachable. Base rate is against it (RE: gemma double-braces).

# STEP 4 — gemma board-faithful: ONLY if STEP 3 broke the cap (else skip — nothing to confirm)
python -m harness.elicit_search --agent gguf --model gemma --guard optimal_public --reps 3 --rounds 6 --budget-s 1800 --max-hops 8 --max-new-tokens 1024
```
Budget note: raw is deterministic under greedy decode; `--reps` denoises LATENCY only. Do NOT grind — if gemma
collapses in STEP 3, stop throughput and redirect to coverage.

### After the session — decide + submit (gated on the above)
- **gemma cap BROKEN (STEP 3 posts>1):** the biggest find in the project. Build a gemma multi-hop sentinel Sub2
  arm; both legs can climb → board 120-135 has a mechanism. (Also confirm STEP 4 survives the public guard.)
- **gemma cap CONFIRMED (STEP 3 all raw=18):** 120 is out of reach by throughput. If STEP 1/2 found a gpt
  multipost winner, fold it into `attack_sub2_multidistinct.py`'s probe set and A/B vs the proven 78; otherwise
  ~88 is the earned ceiling. Then **redirect remaining effort to the collapse-arm coverage** (`attack_sub1_balanced.py`)
  where the actual top-5 ticket lives.

---

## SECONDARY (already run 2026-08-17 GPU #2 — rerun only to re-confirm)
`coverage_leg_probe.py` (gemma C/D/X fire-rate — ANSWERED YES 3/3 both models), `d_suppress_probe.py`
(D-suppress DEAD 0/4), `throughput_bandit.py` (fixed-template; multi_distinct 1.1-1.3x noisy). These are
subsumed by `elicit_search.py` for the throughput fronts; keep for spot-checks.

## Standing reality (unbiased)
Honest top-5 PRIVATE ≈ 10-15%; most-likely outcome mid-pack. Throughput ceiling by KNOWN means ≈ 96-104
(gpt multipost lifts only the gpt leg; gemma parser-capped ~88-95) — **120 needs the gemma cap to break**, which
the base rate is against but which this session measures board-faithfully instead of assuming. Our own gateway
names the private guard `persistent_provenance` (sentinel SURVIVES → provenance world → coverage has no edge,
throughput mid-pack); the host's "public/private don't overlap" tilts toward the collapse world (coverage wins).
Genuinely unresolved → best-of-2 hedge is correct; push BOTH arms. Manually select BOTH finals at close.
