# GPU SESSION RUNBOOK — next pod (both arms, turnkey)

Board-faithful throughout: `--agent gguf` (llama.cpp on the hosted GGUFs; hosted parity = max_new_tokens 1024,
greedy, max-hops 8). `allow_all` guard measures ELICITATION (does the model perform the chain); `optimal_public`
measures public survival. Full pod setup (upload zips, deps, env vars) is in memory **jed-runpod-gpu-workflow**
— deploy any available 24GB+ card (A40 $0.44 / A5000 $0.27), one model at a time.

Ordered by EV toward TOP-5 PRIVATE (from the generative brainstorm wf_e97401b4). Run top-down; stop when the
pod budget runs out — the earlier steps decide the most.

---

## 1. [COVERAGE — highest EV] Does the collapse arm have a GEMMA leg?
The single biggest unknown: gemma fires C 4/4, but D and X are UNMEASURED. A dead gemma D/X leg HALVES the
collapse payoff under the mean. This gates the Sub1 reweight.
```
python -m harness.coverage_leg_probe --agent gguf --model gemma  --reps 3
python -m harness.coverage_leg_probe --agent gguf --model gpt_oss --reps 3   # baseline (expect all fire)
```
**Decision:** if gemma fires D (single fs.write) AND X (2-msg read->post) → collapse arm has a full 2nd leg →
proceed to step 3 to make D cheaper. If gemma-D fires but X doesn't → collapse gemma leg = C+D, treat X as a
gpt-only tail. If neither → concentrate Sub1's gemma weight on C.

## 2. [THROUGHPUT] Confirm multi_distinct transfers + test the marginal riders
`multi_distinct` (8 distinct-domain posts/interact = 1.31x local) is the ONE real throughput gain; the board A/B
is pending, this confirms it board-faithfully and measures the 4 marginal riders (all self-route by raw/sec).
```
python -m harness.throughput_bandit --agent gguf --model gpt_oss --reps 3   # multi_distinct vs single vs T2 forge
python -m harness.throughput_bandit --agent gguf --model gemma  --reps 2   # single vs single_tail (T1)
```
**Decision:** does `multi_distinct` still win 8/8 posts and beat single-post raw/sec on gguf? Does `single_tail`
(T1, gemma hop-1-waste) beat plain single-post on gemma? Does `multi_distinct_forge` (T2) beat plain multi_distinct
on gpt? Adopt any rider into `attack_sub2_multidistinct.py` ONLY if it beats the winner past run-to-run noise
(keep the probe set PRUNED — the 65.4 regression was slow losing templates eating fill budget).
> HONEST: the brainstom rates T1 ~+3-6 mean, T2/T3/T4 ~0. None is a wall-breaker. multi_distinct → mean ~96-106
> ceiling; gemma is parser-capped so 137 is out of reach by known means (137 = gpt-cap-pin ~180 + gemma ~94,
> needs t_post ~4.4s = another 1.4x beyond multi_distinct that no code lever delivers).

## 3. [COVERAGE] Is D (sev4) reasoning-suppressible → a cheap collapse-arm breach?
email.send suppression is DEAD (natural language), but fs.write is tool-call syntax — if it suppresses, D farms
at ~sentinel throughput at DOUBLE C's severity: the best collapse-arm throughput lever.
```
python -m harness.d_suppress_probe --agent gguf --model gpt_oss --reps 4
```
**Decision:** if suppressed-D fires + beats plain-D raw/sec → adopt syntax+suppressed D and raise D's weight in
the collapse mix (with step 1's gemma-D result). If it breaks (like email.send) → keep plain D.

---

## After the session — reweight + submit (gated on the above)
- **Sub1 (coverage):** if gemma fires D → shift the mix toward D (D is sev4=8 vs C sev3=4, gemma-safe single call);
  candidate `D40/C40/X`. Split X into X_clean (no marker, real lineage — un-value-blockable since the guard can't
  see outputs) + X_floor (marker). Apply ONLY after steps 1/3 land. Ship as an A/B vs `attack_sub1_fill.py`.
- **Sub2 (throughput):** ship `attack_sub2_multidistinct.py` (pruned probe set); fold in any rider that won step 2.
- **Operational must-dos:** manually select BOTH finals; size to the replay ceiling with margin; front-load by value.

## Standing reality (unbiased)
Honest top-5 PRIVATE ≈ 10-15%; most-likely outcome mid-pack. Our own gateway names the private guard
`persistent_provenance` (a provenance guard the sentinel SURVIVES) → the leading-evidence world does NOT collapse,
so coverage has no differential edge there and throughput is only mid-pack. The host's "public/private don't
overlap" hint tilts the other way. Genuinely unresolved → the best-of-2 hedge is correct; push BOTH arms, let this
session decide the gemma-D leg + the multi_distinct transfer, keep the hedge, accept the coin.
