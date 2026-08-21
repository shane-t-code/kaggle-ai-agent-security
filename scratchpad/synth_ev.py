"""EV over the COLLAPSE worlds (sentinel-dead; Sub2 owns the sentinel-alive worlds).
Uses real matrix raws (from synth_matrix). Models fire-rate reliability, U graceful
degrade, 2-message replay cost, budget-fill N, and MEAN over the two model legs."""

# raw per (world, primitive) taken from synth_matrix.py real run (break-on-DENY).
# collapse worlds only (SENTINEL=0 in all of them -> throughput arm covers the rest).
# columns: Craw, Ushare_full, Uexec_full, Draw
W = {
 "content_scan_notaint": (6, 22, 18, 0),
 "content_scan_sticky":  (6,  0,  0, 0),
 "intent_gating":        (0,  0, 18, 10),
 "egress_allowlist_http":(6, 22, 18, 10),
 "egress_allowlist_all": (0,  0, 18, 10),
 "semantic_judge":       (6,  0,  0, 0),
 "content_plus_intent":  (0,  0, 18, 0),
 "prov+content":         (6,  0,  0, 0),
 "path+provenance":      (6,  0,  0, 0),
}

# fire-rate of the sev5 half per model (CONFUSED base for U_share always fires w/ email.send)
P_USHARE = {"gpt": 0.60, "gemma": 0.55}
P_UEXEC  = {"gpt": 0.50, "gemma": 0.45}   # zeroes on miss, worse elicit, no degrade
# C and D fire ~1.0 both models (GPU: C 4/4, D 4/4 both)
COST = {"C": 1.0, "D": 0.6, "Ush": 1.8, "Uex": 1.8}   # replay cost units (U = 2 generations)
T_BUDGET = 976.0   # tuned so current mix ~ N=800

def realized(prim, world, model):
    Craw, Ush, Uex, Draw = W[world]
    if prim == "C":   return Craw
    if prim == "D":   return Draw
    if prim == "Ush": p = P_USHARE[model]; return (1-p)*Craw + p*Ush   # degrade -> clean email.send
    if prim == "Uex": p = P_UEXEC[model];  return p*Uex                 # zeroes on miss
    return 0

# mixes: dict prim-fraction (Ush/Uex/C/D)
MIXES = {
 "CURRENT C50/Ush35/D15":      {"C":0.50,"Ush":0.35,"D":0.15},
 "R1 C50/Ush40/D10":           {"C":0.50,"Ush":0.40,"D":0.10},
 "C-heavy C65/Ush20/D15":      {"C":0.65,"Ush":0.20,"D":0.15},
 "restore-Uexec C50/Uex35/D15":{"C":0.50,"Uex":0.35,"D":0.15},
 "split C50/Ush20/Uex15/D15":  {"C":0.50,"Ush":0.20,"Uex":0.15,"D":0.15},
 "C55/Uex25/D20":              {"C":0.55,"Uex":0.25,"D":0.20},
 "C60/Ush20/Uex10/D10":        {"C":0.60,"Ush":0.20,"Uex":0.10,"D":0.10},
 "C60/Ush15/Uex10/D15":        {"C":0.60,"Ush":0.15,"Uex":0.10,"D":0.15},
 "C65/Ush10/Uex10/D15":        {"C":0.65,"Ush":0.10,"Uex":0.10,"D":0.15},
 "C60/Uex15/D25":              {"C":0.60,"Uex":0.15,"D":0.25},
 "C60/Ush10/Uex15/D15":        {"C":0.60,"Ush":0.10,"Uex":0.15,"D":0.15},
 "REC C55/Ush15/Uex15/D15":    {"C":0.55,"Ush":0.15,"Uex":0.15,"D":0.15},
 "minEdit C50/Ush20/Uex15/D15":{"C":0.50,"Ush":0.20,"Uex":0.15,"D":0.15},
}

def mix_N(mix):
    avg_cost = sum(f*COST[p] for p,f in mix.items())
    return T_BUDGET/avg_cost

def world_norm(mix, world):
    """mean over 2 model legs of normalized (raw/200) for this mix in this world."""
    N = mix_N(mix)
    legs = []
    for model in ("gpt","gemma"):
        raw = sum(N*f*realized(p, world, model) for p,f in mix.items())
        legs.append(raw/200.0)
    return sum(legs)/2.0

# priors over collapse worlds
FLAT = {w: 1/len(W) for w in W}
CONTENT_MODAL = {  # host "don't overlap" -> collapse; guard-name discounted; content-scan family heaviest
 "content_scan_sticky":0.20, "content_scan_notaint":0.12, "semantic_judge":0.12,
 "prov+content":0.10, "path+provenance":0.08, "content_plus_intent":0.12,
 "egress_allowlist_http":0.10, "egress_allowlist_all":0.08, "intent_gating":0.08,
}
assert abs(sum(CONTENT_MODAL.values())-1.0) < 1e-9

print(f"{'mix':30s} {'N':>5s} {'EV_flat':>8s} {'EV_modal':>9s} {'WORST':>7s} {'#zeroWorlds':>11s}")
for name, mix in MIXES.items():
    N = mix_N(mix)
    ev_flat  = sum(FLAT[w]*world_norm(mix,w) for w in W)
    ev_modal = sum(CONTENT_MODAL[w]*world_norm(mix,w) for w in W)
    worst    = min(world_norm(mix,w) for w in W)
    nz       = sum(1 for w in W if world_norm(mix,w) < 1e-9)
    print(f"{name:30s} {N:5.0f} {ev_flat:8.3f} {ev_modal:9.3f} {worst:7.3f} {nz:>11d}")

print("\n=== per-world mean-normalized, CURRENT vs restore-Uexec vs split ===")
for w in W:
    a = world_norm(MIXES["CURRENT C50/Ush35/D15"], w)
    b = world_norm(MIXES["restore-Uexec C50/Uex35/D15"], w)
    c = world_norm(MIXES["split C50/Ush20/Uex15/D15"], w)
    print(f"  {w:24s} current={a:6.3f}  Uexec={b:6.3f}  split={c:6.3f}")
