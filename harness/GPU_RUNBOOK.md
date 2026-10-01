# GPU Evaluation Harness — Runbook

The harness replays attack candidates through the real competition SDK against the two target
models on their board-faithful GGUF builds, so throughput and elicitation can be measured
locally before spending a board submission. Any 24 GB+ GPU works (an A40 48 GB is a good value
pick); one model is loaded at a time.

## Pod setup

Use a RunPod PyTorch image with Jupyter. Upload `harness.zip` (this repo's `harness/`, zipped) and the
Kaggle competition data bundle, then from a terminal:

```
cd /workspace
apt-get update -qq && apt-get install -y -qq unzip
unzip -o harness.zip
unzip -o ai-agent-security-multi-step-tool-attacks.zip -d sdk_bundle
ls sdk_bundle    # expect: aicomp_sdk  aicomp_sdk-<ver>.dist-info  kaggle_evaluation  __init__.py
export AICOMP_SDK_BUNDLE=/workspace/sdk_bundle
export PYTHONPATH=/workspace/sdk_bundle:/workspace
export HF_HOME=/workspace/hf
pip install -q pydantic "gymnasium<1" openai huggingface_hub numpy
CMAKE_ARGS="-DGGML_CUDA=on" pip install -q llama-cpp-python --no-cache-dir   # CUDA build, ~5-10 min
python -c "import llama_cpp; print('llama_cpp ok')"
```

Re-export the three environment variables in every new terminal. The GGUF weights auto-download
to `/workspace/hf` on first run (~12 GB gpt-oss, ~17 GB gemma); the unauthenticated-HF warning is
benign. A stop/resume can clear the HF cache and the llama-cpp build — re-run the pip lines if
`import llama_cpp` fails, and budget for a re-download.

## Board-faithful settings

Use `--agent gguf` (llama.cpp on the hosted GGUFs). Hosted parity: `max_new_tokens 1024`, greedy
decoding, 8 tool hops. Two guards select what is measured:

- `allow_all` — elicitation / throughput (does the model perform the chain at all).
- `optimal_public` — survival against the public guardrail.

Greedy decoding is deterministic, so a candidate's scored events reproduce run to run; `--reps`
only denoises wall-clock latency. A local raw/sec win is a candidate for a board A/B, never a
confirmed board gain on its own — multipost throughput has a measured local-to-board gap, so the
board is the only settle.

## Probes

Each probe loads one model, replays through the real scorer, and prints a verdict.

- `elicit_search.py` — two-phase search (UCB1 map → evolutionary mutation) over a template grammar
  (framing × post-count × suppression-tail × argument terseness × URL scheme), reward = real
  raw/sec through the model and scorer; flags winners for a board A/B. Seeds in `elicit_search_seeds.py`.
- `token_cost_probe.py` — decode-tokens-per-scored-post, a hardware-invariant throughput metric;
  `--balloon` reports the continuation-hop balloon rate.
- `cpu_throughput_probe.py` — ranks continuation-suppression phrasings with `n_gpu_layers=0`, since
  CPU decode exposes continuation cost that GPU decode hides.
- `coverage_leg_probe.py` / `sub1_leg_probe.py` — per-primitive fire-rate for the coverage arm on
  both models.
- `gemma_fill_probe.py` — whether the gemma arm fills the replay budget or leaves it on the table.
- `guardrail_zoo.py` / `organizer_guards/` — modeled private-guard variants used as a transfer test.
- `throughput_bandit.py` — fixed-template throughput sweep, kept as a spot-check fallback.

Examples:

```
python -m harness.elicit_search --agent gguf --model gpt_oss --guard allow_all --reps 3 --max-hops 8 --max-new-tokens 1024
python -m harness.token_cost_probe --model gpt_oss --balloon --reps 16
python -m harness.cpu_throughput_probe --reps 4 --k 8 --threads 8
python -m harness.coverage_leg_probe --agent gguf --model gemma --reps 4
```
