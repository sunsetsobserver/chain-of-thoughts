#!/usr/bin/env python3
"""
Program 1 — Tutti cloud → two-chord hit → whisper trio
Time grid: 1 tick = 1/16 note.
Bars: B1=3/4 (12 ticks), B2=2/4 (8 ticks), B3=4/4 (16 ticks).
Bar offsets: B1@0, B2@12, B3@20 ; global end boundary = 36 ticks.

What this script does:
- Auth to DCN.
- LLM Contract A (Bar 1): scale + exact times/durations per instrument.
- Programmatic Bar 2: two quarter tutti chords at absolute ticks 12 and 16 (dur=4 each).
- LLM Contract B (Bar 3): 3 chosen instruments + contrast scale + exact times/durations.
- Deterministic pitch assignment, monophony enforcement, duration capping, dynamics stamping.
- Per-instrument scalar feature encoding (time/duration/pitch/velocity/numerator/denominator).
- One top-level composite referencing all instruments; single execute with N_global.
- Writes `program1_payload.json` (visualiser-ready).
"""

import os, sys, time, json, random, re, requests
from typing import Dict, List, Tuple
from secrets import OPENAI_API_KEY
from eth_account import Account
from eth_account.messages import encode_defunct
from openai import OpenAI

# ---------------- DCN API helpers ----------------
API_BASE = "https://api.decentralised.art"

def _get_account() -> Account:
    priv = os.getenv("PRIVATE_KEY") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if priv:
        acct = Account.from_key(priv); print("Loaded account from PRIVATE_KEY/CLI.")
    else:
        acct = Account.create('KEYSMASH FJAFJKLDSKF7JKFDJ 1530'); print("Created example local account (ephemeral).")
    print(f"Address: {acct.address}")
    return acct

def _handle_response(r: requests.Response):
    try: data = r.json()
    except json.JSONDecodeError:
        r.raise_for_status(); return {"raw": r.text}
    if not r.ok:
        print(f" fail: {r.status_code} {data}", file=sys.stderr)
        raise requests.HTTPError(f" fail: {r.status_code} {data}", response=r)
    return data

def get_nonce(base_url: str, address: str, timeout: float = 10.0) -> str:
    url = f"{base_url}/nonce/{address}"
    r = requests.get(url, headers={"Accept": "application/json"}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "nonce" in data:
        return str(data["nonce"])
    raise ValueError(f"Unexpected nonce response: {data}")

def post_auth(base_url: str, address: str, message: str, signature: str, timeout: float = 10.0) -> dict:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = {"address": address, "message": message, "signature": signature}
    r = requests.post(f"{base_url}/auth", headers=headers, json=payload, timeout=timeout)
    return _handle_response(r)

def post_refresh(base_url: str, access_token: str, refresh_token: str, timeout: float = 10.0) -> dict:
    headers = {
        "Content-Type": "application/json", "Accept": "application/json",
        "Authorization": f"Bearer {access_token}", "X-Refresh-Token": refresh_token,
    }
    r = requests.post(f"{base_url}/refresh", headers=headers, json={}, timeout=timeout)
    return _handle_response(r)

def post_feature(base_url: str, access_token: str, refresh_token: str, payload: dict, timeout: float = 10.0) -> dict:
    headers = {
        "Content-Type": "application/json", "Accept": "application/json",
        "Authorization": f"Bearer {access_token}", "X-Refresh-Token": refresh_token,
    }
    r = requests.post(f"{base_url}/feature", headers=headers, json=payload, timeout=timeout)
    return _handle_response(r)

def post_feature_with_retry(base_url, access, refresh, payload_fn, acct):
    """
    payload_fn(): lambda returning the JSON payload (built just-in-time).
    Returns: (response_json, new_access, new_refresh)
    """
    payload = payload_fn()
    # First try
    try:
        res = post_feature(base_url, access, refresh, payload)
        return res, access, refresh
    except requests.HTTPError as e:
        if getattr(e.response, "status_code", None) != 401:
            raise

    # 401 → try quick refresh
    try:
        ref = post_refresh(base_url, access, refresh)
        access  = ref.get("access_token", access)
        refresh = ref.get("refresh_token", refresh)
        res = post_feature(base_url, access, refresh, payload)
        return res, access, refresh
    except requests.HTTPError as e2:
        if getattr(e2.response, "status_code", None) != 401:
            raise

    # still 401 → full re-auth, then retry once
    nonce = get_nonce(base_url, acct.address)
    msg   = f"Login nonce: {nonce}"
    sig   = acct.sign_message(encode_defunct(text=msg)).signature.hex()
    auth  = post_auth(base_url, acct.address, msg, sig)
    access  = auth.get("access_token", access)
    refresh = auth.get("refresh_token", refresh)
    res = post_feature(base_url, access, refresh, payload)
    return res, access, refresh

def reauth(acct):
    nonce = get_nonce(API_BASE, acct.address)
    msg   = f"Login nonce: {nonce}"
    sig   = acct.sign_message(encode_defunct(text=msg)).signature.hex()
    auth  = post_auth(API_BASE, acct.address, msg, sig)
    new_access  = auth.get("access_token")
    new_refresh = auth.get("refresh_token")
    if not new_access or not new_refresh:
        raise RuntimeError("Re-auth failed")
    return new_access, new_refresh

# ---------------- Lutosławski-style aggregate helpers ----------------
# An aggregate is a dict:
# {
#   "name": "string",
#   "tonic_midi": int,              # optional, kept only as metadata
#   "bands": [
#       {"midi_lo": int, "midi_hi": int, "pcs": [int 0..11 ... 3..4 items, ORDERED]},
#       ...
#   ]
# }
# Rule: pcs are ABSOLUTE pitch-classes (C=0..B=11), and the union across all bands is exactly 12.
# Disjointness: no pitch-class may appear in more than one band.

def agg_validate_and_prepare(agg: dict) -> dict:
    if not isinstance(agg, dict):
        raise RuntimeError("aggregate_scale must be an object.")
    if "bands" not in agg or not isinstance(agg["bands"], list) or not agg["bands"]:
        raise RuntimeError("aggregate_scale.bands must be a non-empty list.")

    used = set()
    norm = []
    for i, b in enumerate(agg["bands"]):
        if not isinstance(b, dict):
            raise RuntimeError(f"aggregate_scale.bands[{i}] must be an object.")
        if not {"midi_lo","midi_hi","pcs"}.issubset(b.keys()):
            raise RuntimeError(f"aggregate_scale.bands[{i}] missing fields (midi_lo, midi_hi, pcs).")
        lo = int(b["midi_lo"]); hi = int(b["midi_hi"])
        if hi < lo: lo, hi = hi, lo
        pcs = [int(x) % 12 for x in b["pcs"]]
        if len(pcs) < 3 or len(pcs) > 4:
            raise RuntimeError(f"aggregate band {i}: pcs length must be 3..4.")
        if len(set(pcs)) != len(pcs):
            raise RuntimeError(f"aggregate band {i}: pcs must be unique within the band.")
        # global disjointness
        if used.intersection(pcs):
            clash = sorted(used.intersection(pcs))
            raise RuntimeError(f"aggregate disjointness violated: pcs {clash} repeated across bands.")
        used.update(pcs)
        norm.append({"midi_lo": lo, "midi_hi": hi, "pcs": pcs})

    if len(used) != 12:
        raise RuntimeError("aggregate must cover all 12 pitch-classes across bands (union size must be 12).")

    norm.sort(key=lambda x: x["midi_lo"])
    out = dict(agg)
    out["bands"] = norm
    out["tonic_midi"] = int(agg.get("tonic_midi", 60))
    return out

def agg_validate_prepare_or_repair(agg: dict) -> dict:
    """
    Try strict validate; if it fails, repair into legal 3–4 pcs bands (disjoint, union=12), then validate again.
    This makes the pipeline robust to imperfect LLM output.
    """
    try:
        return agg_validate_and_prepare(agg)
    except RuntimeError as e_first:
        # --- Mild repair: normalize bands, unique PCs, clip to <=4, and fill to >=3 / union=12 ---
        bands_in = [b for b in (agg.get("bands") or []) if isinstance(b, dict)]
        norm = []
        for b in bands_in:
            lo = int(b.get("midi_lo", 28)); hi = int(b.get("midi_hi", 103))
            if hi < lo: lo, hi = hi, lo
            pcs = [int(x) % 12 for x in (b.get("pcs") or [])]
            # de-dup preserving order, clip to max 4
            seen = set(); pcs = [x for x in pcs if (x not in seen and not seen.add(x))][:4]
            norm.append({"midi_lo": lo, "midi_hi": hi, "pcs": pcs})

        # keep first occurrence of each pc across bands (disjointness)
        seen = set()
        for b in norm:
            newpcs = []
            for pc in b["pcs"]:
                if pc not in seen:
                    newpcs.append(pc); seen.add(pc)
            b["pcs"] = newpcs

        # ensure each band has at least 3 pcs (we'll fill from missing set)
        missing = [pc for pc in range(12) if pc not in seen]
        # first pass: raise bands with <3
        i = 0
        while missing and any(len(b["pcs"]) < 3 for b in norm):
            if len(norm[i]["pcs"]) < 3:
                norm[i]["pcs"].append(missing.pop(0))
            i = (i + 1) % max(1, len(norm))

        # second pass: distribute remaining into bands with <4
        i = 0
        while missing and any(len(b["pcs"]) < 4 for b in norm):
            if len(norm[i]["pcs"]) < 4:
                norm[i]["pcs"].append(missing.pop(0))
            i = (i + 1) % max(1, len(norm))

        # If still missing (rare, e.g. all bands already 4), rebuild into 3 clean bands of 4.
        if missing or not norm:
            # Derive a degree order from whatever the LLM gave; else fallback 0..11.
            order = []
            for b in bands_in:
                for pc in (b.get("pcs") or []):
                    pc = int(pc) % 12
                    if pc not in order:
                        order.append(pc)
            for pc in range(12):
                if pc not in order:
                    order.append(pc)
            order = order[:12]  # ensure exactly 12

            # Three bands of 4 pcs covering low/mid/high orchestra registers
            norm = [
                {"midi_lo": 28, "midi_hi": 47, "pcs": order[0:4]},
                {"midi_lo": 48, "midi_hi": 71, "pcs": order[4:8]},
                {"midi_lo": 72, "midi_hi": 103, "pcs": order[8:12]},
            ]

        repaired = {"name": agg.get("name", "repaired"), "tonic_midi": int(agg.get("tonic_midi", 60)), "bands": norm}
        try:
            return agg_validate_and_prepare(repaired)
        except RuntimeError as e_second:
            # as a last resort, force a deterministic safe split
            safe = {
                "name": "safe-default",
                "tonic_midi": int(agg.get("tonic_midi", 60)),
                "bands": [
                    {"midi_lo": 28, "midi_hi": 47, "pcs": [0, 3, 6, 9]},
                    {"midi_lo": 48, "midi_hi": 71, "pcs": [2, 5, 8, 11]},
                    {"midi_lo": 72, "midi_hi": 103, "pcs": [1, 4, 7, 10]},
                ],
            }
            return agg_validate_and_prepare(safe)
        
def pc_to_band_map(agg: dict) -> Dict[int, int]:
    """Map each pc 0..11 to its band index."""
    m = {}
    for i, b in enumerate(agg["bands"]):
        for pc in b["pcs"]:
            m[int(pc) % 12] = i
    return m

def aggregate_similarity(a: dict, b: dict) -> float:
    """
    Fraction of pcs mapped to the same band in both aggregates (0..1).
    1.0 = identical pc→band partition; 0.0 = completely different.
    """
    ma, mb = pc_to_band_map(a), pc_to_band_map(b)
    same = sum(1 for pc in range(12) if ma.get(pc) == mb.get(pc))
    return same / 12.0

def has_equal_step_band(agg: dict) -> bool:
    """
    True if any band is a pure equal-step cycle (e.g., 0,3,6,9 or 0,4,8).
    We check the ORDER they give (we don't sort; order matters for degree steps).
    """
    for b in agg["bands"]:
        pcs = [int(x) % 12 for x in b["pcs"]]
        if len(pcs) >= 3:
            steps = [ (pcs[(i+1)%len(pcs)] - pcs[i]) % 12 for i in range(len(pcs)) ]
            if len(set(steps)) == 1:
                return True
    return False

def agg_find_band_for_midi(m: int, bands: List[dict]) -> int:
    # choose the containing band nearest to its center; if none contain, choose nearest by range distance
    containing = [(i, b) for i, b in enumerate(bands) if b["midi_lo"] <= m <= b["midi_hi"]]
    if containing:
        i, _ = min(containing, key=lambda ib: abs(m - ((ib[1]["midi_lo"] + ib[1]["midi_hi"]) // 2)))
        return i
    def dist_to_range(b):
        if m < b["midi_lo"]: return b["midi_lo"] - m
        if m > b["midi_hi"]: return m - b["midi_hi"]
        return 0
    i, _ = min(enumerate(bands), key=lambda ib: dist_to_range(ib[1]))
    return i

def agg_allowed_pcs_for_midi(m: int, agg: dict) -> set:
    bands = agg["bands"]
    i = agg_find_band_for_midi(m, bands)
    return set(bands[i]["pcs"])

def agg_global_degree_order(agg: dict) -> List[int]:
    # ordered by scanning bands from low to high and appending pcs in each band's given order
    out = []
    for b in agg["bands"]:
        for pc in b["pcs"]:
            if pc not in out:
                out.append(pc)
    if len(out) != 12:
        # shouldn't happen after validate, but be defensive
        rest = [pc for pc in range(12) if pc not in out]
        out.extend(rest)
    return out

def nearest_pitch_for_pc_aggregate(target_pc: int, approx: int, rng: Tuple[int,int], agg: dict) -> int:
    lo, hi = rng
    best = None; bestd = 10**9
    for delta in range(0, 49):  # search radius
        for sgn in (+1, -1):
            cand = clamp(approx + sgn*delta, lo, hi)
            if cand % 12 != target_pc:
                continue
            # allowed only if the band containing this MIDI includes the target_pc
            bands = agg["bands"]
            bi = agg_find_band_for_midi(cand, bands)
            if target_pc in bands[bi]["pcs"]:
                d = abs(cand - approx)
                if d < bestd:
                    best, bestd = cand, d
        if best is not None and delta > 12:  # early exit once something is reasonably close
            break
    if best is None:
        # last resort: snap to any allowed pc at approx position
        allowed = agg_allowed_pcs_for_midi(approx, agg)
        for delta in range(0, 49):
            for sgn in (+1, -1):
                cand = clamp(approx + sgn*delta, lo, hi)
                if (cand % 12) in allowed:
                    return cand
        return clamp(approx, lo, hi)
    return best

def pitches_from_steps_aggregate(times: List[int], steps: List[int], agg: dict,
                                 rng: Tuple[int, int], tess: Tuple[int, int],
                                 start_hint: str) -> List[int]:
    if not times:
        return []
    lo, hi = rng
    cen = (tess[0] + tess[1]) // 2
    if start_hint == "low":
        guess = max(lo, cen - 7)
    elif start_hint == "high":
        guess = min(hi, cen + 7)
    else:
        guess = cen

    # choose start: nearest allowed pitch at guess according to aggregate
    bi0 = agg_find_band_for_midi(guess, agg["bands"])
    pcs0 = agg["bands"][bi0]["pcs"]
    start = None; bestd = 10**9
    for pc in pcs0:
        cand = nearest_pitch_for_pc_aggregate(pc, guess, rng, agg)
        d = abs(cand - guess)
        if d < bestd:
            start, bestd = cand, d
    if start is None:
        start = clamp(guess, lo, hi)

    out = [start]
    need = max(0, len(times) - 1)
    s = (steps[:need] + [0]*need)[:need]

    for st in s:
        cur = out[-1]
        bi = agg_find_band_for_midi(cur, agg["bands"])
        band_pcs = agg["bands"][bi]["pcs"]  # ORDERED list for degree stepping
        cur_pc = cur % 12
        if cur_pc not in band_pcs:
            # snap current to nearest allowed pc in this band
            cand = nearest_pitch_for_pc_aggregate(band_pcs[0], cur, rng, agg)
            cur = cand
            out[-1] = cand
            cur_pc = cur % 12
        cur_idx = band_pcs.index(cur_pc)
        new_idx = (cur_idx + int(st)) % len(band_pcs)
        target_pc = band_pcs[new_idx]
        nxt = nearest_pitch_for_pc_aggregate(target_pc, cur, rng, agg)
        out.append(nxt)

    return out

def scale_walk_assign_pitches_aggregate(times: List[int], agg: dict, rng: Tuple[int,int], prefer_center: Tuple[int,int]) -> List[int]:
    if not times:
        return []
    lo, hi = rng
    center = clamp((prefer_center[0] + prefer_center[1]) // 2, lo, hi)
    # deterministic small pattern, but realized inside aggregate
    pattern = [1, 1, -2, 1, -1, 2]

    # choose start nearest allowed at center
    bi0 = agg_find_band_for_midi(center, agg["bands"])
    pcs0 = agg["bands"][bi0]["pcs"]
    start = nearest_pitch_for_pc_aggregate(pcs0[0], center, rng, agg)
    out = [start]

    k = 0
    for _ in range(1, len(times)):
        cur = out[-1]
        bi = agg_find_band_for_midi(cur, agg["bands"])
        pcs = agg["bands"][bi]["pcs"]
        cur_pc = cur % 12
        if cur_pc not in pcs:
            # resnap
            cur = nearest_pitch_for_pc_aggregate(pcs[0], cur, rng, agg)
            out[-1] = cur
            cur_pc = cur % 12
        st = pattern[k % len(pattern)]
        k += 1
        new_idx = (pcs.index(cur_pc) + st) % len(pcs)
        target_pc = pcs[new_idx]
        nxt = nearest_pitch_for_pc_aggregate(target_pc, cur, rng, agg)
        out.append(nxt)
    return out



# ---------------- Musical constants ----------------
# Tick math
TICKS_PER_16TH = 1
BAR1_TICKS = 12  # 3/4
BAR2_TICKS = 8   # 2/4
BAR3_TICKS = 16  # 4/4
BAR4_TICKS = 12  # 3/4
BAR5_TICKS = 16  # 4/4
BAR6_TICKS = 16  # 4/4
BAR7_TICKS = 12
BAR8_TICKS = 12
BAR9_TICKS = 12

BAR1_OFF, BAR2_OFF, BAR3_OFF = 0, 12, 20
BAR4_OFF, BAR5_OFF, BAR6_OFF = 36, 48, 64 
BAR7_OFF, BAR8_OFF, BAR9_OFF = 80, 92, 104

GLOBAL_END = 116 

# Ranges (sounding MIDI)
INSTRUMENTS = {
    "alto_flute":    {"range": (55, 93),  "tess": (60, 86)},
    "violin":        {"range": (55, 103), "tess": (60, 96)},
    "bass_clarinet": {"range": (34, 82),  "tess": (38, 76)},
    "trumpet":       {"range": (58, 94),  "tess": (62, 88)},
    "cello":         {"range": (36, 76),  "tess": (41, 72)},
    "double_bass":   {"range": (28, 64),  "tess": (31, 57)},
}
ORDERED_INSTRS = ["alto_flute","violin","bass_clarinet","trumpet","cello","double_bass"]

# Dynamics (fixed)
def vel_bar1_p_cresc(local_t: int) -> int:
    # 50 → 80 over ticks 0..11
    return max(0, min(127, round(50 + (80-50) * (local_t / 11.0))))

VEL_BAR2_FIRST = 124
VEL_BAR2_SECOND = 114
VEL_BAR3_PP = 30

# ---------------- Utility funcs ----------------
def request_fresh_aggregate(make_prompts_fn, key_in_json: str,
                            client: OpenAI, prior_aggs: List[dict],
                            attempts: int = 4, temp: float = 1.15) -> Tuple[dict, dict]:
    """
    make_prompts_fn() -> (system, user) already contextualized with diversity notes.
    key_in_json: the field name where the aggregate lives in the returned JSON
                 ("aggregate_scale" or "contrast_aggregate_scale").
    Returns: (full_json, validated_agg)
    """
    best = None; best_agg = None; best_sim = 1.0
    for _ in range(max(1, attempts)):
        sys_msg, usr_msg = make_prompts_fn()
        js = call_llm_json(client=client, system_msg=sys_msg, user_msg=usr_msg,
                           model="gpt-4.1", temperature=temp, top_p=0.9)
        raw = js.get(key_in_json, {}) or {}
        agg = agg_validate_prepare_or_repair(raw)

        # reject equal-step clichés and near-clones to any prior
        sim = max((aggregate_similarity(agg, p) for p in prior_aggs), default=0.0)
        if not has_equal_step_band(agg) and sim <= 0.45 and agg.get("name","") != "safe-default":
            return js, agg  # fresh enough

        # keep the best (lowest similarity) in case all attempts fail
        if sim < best_sim:
            best, best_agg, best_sim = js, agg, sim

    # all attempts exceeded threshold: return the least-similar candidate
    return best or js, best_agg or agg

def clamp(x, lo, hi): return max(lo, min(hi, int(x)))

def _extract_first_json_obj(text: str) -> str:
    if not text: return ""
    start = text.find("{")
    if start == -1: return ""
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return ""

def call_llm_json(client: OpenAI, system_msg: str, user_msg: str,
                  model="gpt-4.1", temperature=0.2, **kwargs) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system_msg},
                  {"role":"user","content":user_msg}],
        temperature=temperature,
        response_format={"type":"json_object"},
        **kwargs
    )
    content = getattr(resp.choices[0].message, "content", "") or ""
    js = _extract_first_json_obj(content)
    if not js:
        raise RuntimeError(f"LLM returned no JSON. Raw: {content[:200]!r}")
    try:
        return json.loads(js)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON: {js[:200]!r}") from e

def enforce_monophony_times(times: List[int]) -> List[int]:
    """Ensure strictly increasing times (drop duplicates)."""
    if not times: return []
    times = sorted(int(t) for t in times)
    out = []
    last = None
    for t in times:
        if last is None or t > last:
            out.append(t); last = t
        # equal => drop
    return out

def cap_durations_local(times: List[int], durs: List[int], bar_len: int) -> Tuple[List[int], List[int]]:
    """Cap durations not to overlap next onset and not to exceed bar boundary."""
    out_t, out_d = [], []
    L = min(len(times), len(durs))
    for i in range(L):
        t = int(times[i]); d = max(1, int(durs[i]))
        # cap to bar
        d = min(d, bar_len - t)
        # cap to next onset
        if i < L-1:
            gap = max(0, int(times[i+1]) - t)
            d = min(d, gap)
        if d > 0:
            out_t.append(t); out_d.append(d)
    return out_t, out_d

def absolute_times(local_times: List[int], bar_offset: int) -> List[int]:
    return [int(t) + bar_offset for t in local_times]

def deltas_from_series(vals: List[int]) -> List[dict]:
    """Encode as add/sub ops from seed=0 so cumulative sum yields vals."""
    ops = []
    prev = 0
    for v in vals:
        delta = int(v) - int(prev)
        if delta >= 0:
            ops.append({"name":"add","args":[delta]})
        else:
            ops.append({"name":"subtract","args":[-delta]})
        prev = v
    # ensure at least one op
    if not ops:
        ops = [{"name":"add","args":[0]}]
    return ops

def _meter_at_time(t: int) -> Tuple[int, int]:
    if t < 12:      return 3, 4   # bar 1 (0..11)
    elif t < 20:    return 2, 4   # bar 2 (12..19)
    elif t < 36:    return 4, 4   # bar 3 (20..35)
    elif t < 48:    return 3, 4   # bar 4 (36..47)
    elif t < 64:    return 4, 4   # bar 5 (48..63)
    elif t < 80:    return 4, 4   # bar 6 (64..79)
    elif t < 92:    return 3, 4   # bar 7 (80..91)
    elif t < 104:   return 3, 4   # bar 8 (92..103)
    elif t < 116:   return 3, 4   # bar 9 (104..115)
    else:           return 3, 4

def make_meter_arrays(N: int, abs_times: List[int]) -> Tuple[List[int], List[int]]:
    """Map each onset time to its bar's meter."""
    nums, dens = [], []
    for t in abs_times:
        n, d = _meter_at_time(t)
        nums.append(n); dens.append(d)
    if len(nums) != N:
        nums = (nums + [nums[-1]]*N)[:N]
        dens = (dens + [4]*N)[:N]
    return nums, dens

def bar2_two_chords_from_aggregate(agg: dict, plan: dict, seed: int = None) -> Dict[str, List[int]]:
    """
    Realize two quarter-note verticals at absolute onsets 12 and 16 (dur=4 each),
    using an LLM-provided harmonic plan over the aggregate's global degree order.
    """
    rng = random.Random(seed)
    degree_order = agg_global_degree_order(agg)  # list of 12 PCs

    deg_roots = plan.get("degree_roots", [0, 4])
    kinds     = plan.get("chord_kinds", ["triad", "triad"])
    invs      = plan.get("inversions", [0, 0])
    spread    = plan.get("spread_hint", "close")
    avoid_plan= plan.get("avoid_plan", "same")

    def build_chord(root_idx: int, kind: str) -> List[int]:
        skip = [0,2,4] if kind == "triad" else [0,2,4,6]
        return [degree_order[(root_idx + s) % len(degree_order)] for s in skip]

    chord1_pcs_ord = _invert_pcset(build_chord(deg_roots[0] % 12, kinds[0]), invs[0])
    chord2_pcs_ord = _invert_pcset(build_chord(deg_roots[1] % 12, kinds[1]), invs[1])

    if avoid_plan == "same" and sorted(chord1_pcs_ord) == sorted(chord2_pcs_ord):
        deg_roots[1] = (deg_roots[1] + 1) % 12
        chord2_pcs_ord = _invert_pcset(build_chord(deg_roots[1], kinds[1]), invs[1])

    centers = {
        "double_bass": -12, "cello": -7, "bass_clarinet": -3,
        "trumpet": +2, "violin": +7, "alto_flute": +9,
    }
    widen = 5 if spread == "open" else 0

    abs_times = [12, 16]
    durs = [4, 4]
    voices = {}

    prefer_idx_map = {
        "double_bass": 0,
        "cello":       1 if len(chord1_pcs_ord) > 2 else 0,
        "bass_clarinet": 2 if len(chord1_pcs_ord) > 2 else 1,
        "trumpet":     -1,
        "violin":      -1,
        "alto_flute":  -1,
    }

    for instr in ORDERED_INSTRS:
        lo, hi = INSTRUMENTS[instr]["range"]
        tess_lo, tess_hi = INSTRUMENTS[instr]["tess"]
        center_guess = (tess_lo + tess_hi)//2 + centers.get(instr, 0)

        idx_pref = prefer_idx_map[instr]
        i1 = (idx_pref if idx_pref != -1 else rng.randrange(len(chord1_pcs_ord))) % len(chord1_pcs_ord)
        pc1 = chord1_pcs_ord[i1]
        p1  = nearest_pitch_for_pc_aggregate(pc1, center_guess - widen, (lo, hi), agg)

        pc2_pref = chord2_pcs_ord[i1 % len(chord2_pcs_ord)]
        p2  = nearest_pitch_for_pc_aggregate(pc2_pref, p1 + rng.choice([-2,-1,0,1,2]), (lo, hi), agg)
        if p2 == p1:
            alt_pc2 = rng.choice(chord2_pcs_ord)
            p2 = nearest_pitch_for_pc_aggregate(alt_pc2, p1 + rng.choice([-3,-2,-1,1,2,3]), (lo, hi), agg)

        voices[instr] = {
            "time": abs_times[:],
            "duration": durs[:],
            "pitch": [p1, p2],
            "velocity": [VEL_BAR2_FIRST, VEL_BAR2_SECOND],
        }
    return voices


# ---------------- LLM Contracts ----------------
def prompt_bar1_contract() -> Tuple[str, str]:
    system = """
You output ONLY strict JSON. No prose.

Goal: Compose Bar 1 (3/4, 12 ticks @ 1/16) for SIX monophonic instruments, AND define a Lutosławski-style AGGREGATE (register-banded).

Return JSON with EXACTLY this shape:
{
  "aggregate_scale": {
    "name": "string",
    "tonic_midi": int,
    "bands": [
      {"midi_lo": int, "midi_hi": int, "pcs": [int,int,int(,int)]},
      ...
    ]
  },
  "rhythm": {
    "<instr>": {"time":[ints 0..11 strictly increasing], "duration":[positive ints, same length]},
    ...
  },
  "melody": {
    "<instr>": {"start_hint":"low|mid|high", "steps":[ len(time)-1 ints (can be negative/zero) ]},
    ...
  },
  "meta": { "bands_count": int, "band_lengths": [ints], "union_size": int }
}

MANDATORY AGGREGATE RULES (hard constraints):
- pcs are ABSOLUTE pitch-classes (C=0..B=11).
- Each band's pcs length is EXACTLY 3 or 4.
- Bands’ pcs are pairwise DISJOINT (no pc appears in more than one band).
- The UNION across all bands is EXACTLY 12 distinct pcs.
- Bands may overlap in MIDI ranges; only pcs must be disjoint.
- Keep the pcs ORDER within each band meaningful (used for degree stepping).

Bar 1 material rules:
- Instruments: alto_flute, violin, bass_clarinet, trumpet, cello, double_bass (all six must appear).
- Monophony per instrument: times strictly increasing; time[i] + duration[i] <= 12.
- "steps" are DEGREE steps within the ORDERED pcs list of the CURRENT band of the note.

Validation checklist (you MUST satisfy before returning JSON):
1) Every band has 3 or 4 pcs. 2) No duplicate pcs across bands. 3) Union size = 12.
4) For each instrument: len(steps) == max(0, len(time)-1), arrays align, times strictly increasing, durations positive and in range.
If any check fails, regenerate internally and only return a valid JSON object.
"""
    user = """
Instruments: alto_flute, violin, bass_clarinet, trumpet, cello, double_bass.
Soft target onset counts: AF 6..10, Vn 5..9, BCl 3..6, Tpt 2..5, Vc 2..5, Db 2..4.
Return VALID JSON ONLY (no comments, no prose).
"""
    return system.strip(), user.strip()



def prompt_bar2_harmony_contract(b1_agg: dict) -> Tuple[str, str]:
    degree_order = agg_global_degree_order(b1_agg)
    system = """
You output ONLY strict JSON. No prose.

Task: For Bar 2 (2/4; two quarter-note hits at absolute ticks 12 and 16), choose a harmonic plan OVER THE PROVIDED DEGREE_ORDER
(derived from the Bar 1 aggregate).

Return:
{
  "degree_roots": [int, int],              // roots are indices 0..11 into DEGREE_ORDER
  "chord_kinds": ["triad|seventh","triad|seventh"],
  "inversions": [int, int],                // 0..3 (3 valid only for seventh)
  "spread_hint": "close|open",
  "avoid_plan": "same|allow"
}

Checks (you must satisfy): arrays length=2; integers in range; strings in allowed set.
"""
    user = f"""
DEGREE_ORDER (indices 0..11 -> absolute pcs): {degree_order}
Aim for contrast between the two verticals; avoid identical pc-multisets unless "avoid_plan" == "allow".
Return VALID JSON ONLY.
"""
    return system.strip(), user.strip()



def prompt_bar3_contract(b1_agg: dict) -> Tuple[str, str]:
    system = """
    You output ONLY strict JSON. No prose.

    Bar 3 (4/4, 16 ticks): choose EXACTLY three instruments, define a NEW CONTRAST aggregate (same banded schema, NOT reused from Bar 1),
    and provide rhythms + melodic directives for those three instruments.

    Return:
    {
    "chosen_instruments": ["<instr>","<instr>","<instr>"],
    "contrast_aggregate_scale": {
        "name":"string","tonic_midi":int,
        "bands":[{"midi_lo":int,"midi_hi":int,"pcs":[int,int,int(,int)]}, ...]
    },
    "rhythm": { "<instr>": {"time":[ints 0..15 inc], "duration":[>0, same len]} , ... },
    "melody": { "<instr>": {"start_hint":"low|mid|high", "steps":[len(time)-1 ints]} , ... },
    "meta": { "bands_count": int, "band_lengths": [ints], "union_size": int }
    }

    Aggregate constraints (hard):
    - Different from Bar 1’s aggregate in pcs and/or band ranges (materially different, not just renamed).
    - Each band's pcs length is 3 or 4; bands disjoint; union across bands is exactly 12.
    - pcs are ABSOLUTE 0..11; ORDER within each band matters.

    Material constraints:
    - Exactly 3 distinct instruments from: ['alto_flute','violin','bass_clarinet','trumpet','cello','double_bass'].
    - For each chosen instrument: monophony; time[i] + duration[i] <= 16; len(steps) == len(time)-1.
    Validate all constraints; if any fail, regenerate internally before returning JSON.

    Diversity rules vs Bar 1 aggregate (hard):
    - Provide a materially different pc→band mapping, not just a transposition or band rename.
    - At least 6 pitch-classes MUST move to a different band than in Bar 1.
    - Avoid pure equal-step bands (no [0,3,6,9] rotations; no [0,4,8] rotations). At most one band may be an equal-step cycle, but prefer none.
    - Change band MIDI ranges meaningfully (different boundaries and/or overlaps).

    """

    prev_map = pc_to_band_map(b1_agg)
    user = "Return VALID JSON ONLY.\nPrev pc→band map for Bar 1 (for diversity): " + json.dumps(prev_map)

    return system.strip(), user.strip()



def prompt_bars456_pcsets_contract(b1_agg: dict, harm2: dict, bar3: dict) -> Tuple[str, str]:
    deg = harm2.get("degree_roots", [])
    kinds = harm2.get("chord_kinds", [])
    invs = harm2.get("inversions", [])
    bar2_summary = {"degree_roots": deg, "chord_kinds": kinds, "inversions": invs}

    system = """
    You output ONLY strict JSON. No prose.

    Task: Define a NEW aggregate for Bars 4–6 (NOT reused from Bars 1 or 3), and propose pitch-class sets for simultaneities with hints.

    Return:
    {
    "aggregate_scale": {
        "name":"string","tonic_midi":int,
        "bands":[{"midi_lo":int,"midi_hi":int,"pcs":[int,int,int(,int)]}, ...]
    },
    "bar4": {
        "pcset": [int...], "voicing_hint": "cluster|spread",
        "register_targets": {
        "alto_flute":"low|mid|high","violin":"low|mid|high","bass_clarinet":"low|mid|high",
        "trumpet":"low|mid|high","cello":"low|mid|high","double_bass":"low|mid|high"
        },
        "avoid_unisons": "mild|strong"
    },
    "bar5": { ... same shape ... },
    "bar6": [ { ... first half ... }, { ... second half ... } ],
    "meta": { "bands_count": int, "band_lengths": [ints], "union_size": int }
    }

    Aggregate constraints (hard):
    - Different from Bar 1 and Bar 3 aggregates in pcs and/or ranges.
    - Each band's pcs length is 3 or 4; bands disjoint; union size == 12; pcs are 0..11.

    Chord pcset constraints:
    - Each pcset length 3..12, ints 0..11. Hints are advisory; realization is programmatic.

    Validate checklist before returning: aggregate passes all rules; pcsets are valid arrays in range.
    If anything fails, regenerate internally and return only valid JSON.

    Diversity rules vs Bars 1 and 3 aggregates (hard):
    - At least 6 pitch-classes MUST change their band assignment compared to EACH of Bar 1 and Bar 3.
    - Avoid pure equal-step bands (prefer none).
    - Use different band ranges (shift boundaries/overlaps).

    """
    map1 = pc_to_band_map(b1_agg)
    cagg = bar3.get("contrast_aggregate_scale", {})
    map3 = pc_to_band_map(cagg) if isinstance(cagg, dict) and "bands" in cagg else {}
    user = f"""
    Context reminder — Bar 2 plan: {bar2_summary}
    Prev pc→band maps to diverge from:
    - Bar1: {json.dumps(map1)}
    - Bar3: {json.dumps(map3)}
    Return VALID JSON ONLY.
    """.strip()
    return system.strip(), user.strip()



def prompt_bars789_clouds_contract(b1_agg: dict, harm2: dict, bar3: dict, pc456: dict) -> Tuple[str, str]:
    chosen = bar3.get("chosen_instruments", [])

    system = """
    You output ONLY strict JSON. No prose.

    Task: Bars 7–9 (each 3/4, 12 ticks) as *polyphonic clouds* using instrument PAIRS (3 bars = 3 distinct pairs covering all 6 instruments exactly once).
    Define a NEW aggregate for Bars 7–9 (NOT reused from Bars 1, 3, or 4–6).

    Return:
    {
    "aggregate_scale": {
        "name":"string","tonic_midi":int,
        "bands":[{"midi_lo":int,"midi_hi":int,"pcs":[int,int,int(,int)]}, ...]
    },
    "style": "long|staccato",
    "pairs": [["instrA","instrB"], ["instrC","instrD"], ["instrE","instrF"]],
    "bars": [
        { "pair_index": 0,
        "rhythm": { "<instr>": {"time":[ints 0..11 inc], "duration":[>0, same len]} , "<instr>": {...} },
        "melody": { "<instr>": {"start_hint":"low|mid|high","steps":[len(time)-1 ints]}, "<instr>": {...} }
        },
        { "pair_index": 1, ... },
        { "pair_index": 2, ... }
    ],
    "meta": { "bands_count": int, "band_lengths": [ints], "union_size": int }
    }

    Aggregate constraints (hard):
    - Different from Bar 1, Bar 3, and Bars 4–6 aggregates in pcs and/or ranges.
    - Each band has 3 or 4 pcs; bands disjoint; union across bands == 12; pcs are 0..11.

    Pairing constraints:
    - `pairs` must be 3 disjoint pairs covering exactly these six once: ["alto_flute","violin","bass_clarinet","trumpet","cello","double_bass"].

    Per active instrument per bar:
    - If "staccato": 6–10 onsets (durations 1–2). If "long": 4–7 onsets (durations mostly 3–6, rare 7–10).
    - Interlock: avoid >~1/3 aligned onsets; contrast contours; modest register separation.
    - Monophony; time[i] + duration[i] <= 12; len(steps) == len(time)-1.

    Validation checklist (must satisfy before returning):
    1) Aggregate passes all hard rules. 2) Pairs cover all 6 once. 3) For each active instrument: arrays align, times strictly increasing, durations valid, steps length matches.
    If anything fails, regenerate internally and return only valid JSON.

    Diversity rules vs Bars 1, 3, and 4–6 aggregates (hard):
    - At least 6 pitch-classes MUST change band assignment compared to EACH prior section.
    - Avoid pure equal-step bands (prefer none).
    - Use different band ranges than earlier sections.
    """

    map1 = pc_to_band_map(b1_agg)
    cagg = bar3.get("contrast_aggregate_scale", {})
    map3 = pc_to_band_map(cagg) if isinstance(cagg, dict) and "bands" in cagg else {}
    a456 = pc456.get("aggregate_scale", {})
    map456 = pc_to_band_map(a456) if isinstance(a456, dict) and "bands" in a456 else {}
    user = f"""
    Bar 3 chose: {chosen}
    Prev pc→band maps to diverge from:
    - Bar1: {json.dumps(map1)}
    - Bar3: {json.dumps(map3)}
    - Bars4–6: {json.dumps(map456)}
    Return VALID JSON ONLY.
    """.strip()

    return system.strip(), user.strip()



def realize_bars789_clouds(plan: dict) -> Dict[str, Dict[str, List[int]]]:
    """
    Realize Bars 7–9 from the LLM cloud plan using an aggregate scale.
    """
    out = {instr: {"time": [], "duration": [], "pitch": [], "velocity": []} for instr in ORDERED_INSTRS}

    agg_raw = plan.get("aggregate_scale", {})
    if not isinstance(agg_raw, dict):
        agg_raw = {}
    agg = agg_validate_prepare_or_repair(agg_raw)

    pairs = plan.get("pairs", [])
    bars  = plan.get("bars", [])
    offsets = [BAR7_OFF, BAR8_OFF, BAR9_OFF]
    ticks   = [BAR7_TICKS, BAR8_TICKS, BAR9_TICKS]

    for bi in range(min(3, len(bars))):
        bar = bars[bi]
        pi = int(bar.get("pair_index", bi))
        if not (isinstance(pairs, list) and 0 <= pi < len(pairs) and isinstance(pairs[pi], list) and len(pairs[pi])==2):
            continue
        active = [x for x in pairs[pi] if x in ORDERED_INSTRS]
        if len(active) != 2:
            continue

        off = offsets[bi]; L = ticks[bi]
        rhythm = bar.get("rhythm", {})
        melody = bar.get("melody", {})

        for instr in active:
            t_local = rhythm.get(instr, {}).get("time", [])
            d_local = rhythm.get(instr, {}).get("duration", [])
            t_local = enforce_monophony_times([int(x) for x in t_local if 0 <= int(x) < L])
            t_local, d_local = cap_durations_local(t_local, [int(x) for x in d_local], L)

            t_abs = absolute_times(t_local, off)
            rng = INSTRUMENTS[instr]["range"]; tess = INSTRUMENTS[instr]["tess"]
            m = melody.get(instr, {})
            steps = m.get("steps", [])
            start_hint = m.get("start_hint", "mid")

            if isinstance(steps, list) and len(steps) == max(0, len(t_local) - 1):
                p = pitches_from_steps_aggregate(t_local, steps, agg, rng, tess, start_hint)
            else:
                p = scale_walk_assign_pitches_aggregate(t_local, agg, rng, tess)

            v = [VEL_BAR3_PP for _ in t_local]
            out[instr]["time"]    += t_abs
            out[instr]["duration"]+= d_local
            out[instr]["pitch"]   += p
            out[instr]["velocity"]+= v

    return out


# ---------------- Bar 2 chord voicing ----------------

def _chord_pcset(scale_pcs: List[int], deg_root: int, kind: str) -> List[int]:
    """Return ordered chord degrees (as pitch-classes) from ordered scale pcs."""
    degs = len(scale_pcs)
    # triad: 0,2,4 ; seventh: 0,2,4,6 (by scale-degree skips)
    tpl = [0,2,4] if kind == "triad" else [0,2,4,6]
    pcs = [scale_pcs[(deg_root + d) % degs] for d in tpl]
    return pcs  # ordered for inversion handling

def _invert_pcset(pcs_ordered: List[int], inversion: int) -> List[int]:
    """Rotate chord degrees; inversion counted in degrees, not semitones."""
    if not pcs_ordered:
        return pcs_ordered
    k = inversion % len(pcs_ordered)
    return pcs_ordered[k:] + pcs_ordered[:k]

def _nearest_pitch_for_pcset(target_pc: int, approx: int, rng: Tuple[int,int]) -> int:
    """Nearest in-range pitch with given pitch-class."""
    lo, hi = rng
    best = approx; bestd = 10**9
    for k in range(-24, 25):
        cand = clamp(approx + k, lo, hi)
        if cand % 12 == target_pc:
            d = abs(cand - approx)
            if d < bestd:
                best, bestd = cand, d
    return best

def _center_from_register_hint(tess: Tuple[int,int], hint: str) -> int:
    lo, hi = tess
    mid = (lo + hi)//2
    if hint == "low":  return max(lo, mid - 7)
    if hint == "high": return min(hi, mid + 7)
    return mid

def _spread_bias(instr: str, base_center: int, voicing_hint: str) -> int:
    # small deterministic offsets by chair; widen if "spread"
    base = {"double_bass": -12, "cello": -7, "bass_clarinet": -3, "trumpet": +2, "violin": +7, "alto_flute": +9}.get(instr, 0)
    if voicing_hint == "spread":
        base *= 1  # keep simple; multiplier could be >1 for wider spacing
    else:
        base = int(base * 0.5)
    return base_center + base

def realize_pcset_event(pcset: List[int], voicing_hint: str, reg_targets: Dict[str,str], avoid_unisons: str,
                        prev_pitches: Dict[str,int] = None,
                        agg: dict = None) -> Dict[str,int]:
    """
    Choose one pitch per instrument whose pitch-class is in pcset, using range/tessitura and hints.
    If 'agg' is provided, enforce Lutosławski-style band constraints: the chosen pitch must fall in a band
    that contains its pitch-class.
    Returns: {instr: pitch}
    """
    out = {}
    taken = set()
    for instr in ORDERED_INSTRS:
        lo, hi = INSTRUMENTS[instr]["range"]
        tess = INSTRUMENTS[instr]["tess"]
        hint = (reg_targets or {}).get(instr, "mid")
        center = _center_from_register_hint(tess, hint)
        approx = _spread_bias(instr, center, voicing_hint)

        # prefer keep common tone / small move if prev exists
        target_pc = None
        if prev_pitches and instr in prev_pitches and (prev_pitches[instr] % 12) in pcset:
            target_pc = prev_pitches[instr] % 12
        else:
            options = list(pcset)
            if avoid_unisons == "strong" and taken:
                unused = [pc for pc in options if pc not in taken]
                if unused:
                    options = unused
            target_pc = options[0] if options else (approx % 12)

        # choose nearest pitch honoring aggregate if provided
        if agg is not None:
            pitch_pick = nearest_pitch_for_pc_aggregate(int(target_pc), approx, (lo, hi), agg)
        else:
            pitch_pick = _nearest_pitch_for_pcset(int(target_pc), approx, (lo, hi))

        # if still colliding exactly and avoid strong, try another pc
        if avoid_unisons == "strong" and pitch_pick in out.values() and len(pcset) > 1:
            for alt_pc in pcset:
                if alt_pc == target_pc:
                    continue
                if agg is not None:
                    q = nearest_pitch_for_pc_aggregate(int(alt_pc), approx, (lo, hi), agg)
                else:
                    q = _nearest_pitch_for_pcset(int(alt_pc), approx, (lo, hi))
                if q not in out.values():
                    pitch_pick = q
                    target_pc = alt_pc
                    break

        out[instr] = int(pitch_pick)
        taken.add(int(target_pc))
    return out


def realize_bars456(pc_plan: dict) -> Dict[str, Dict[str, List[int]]]:
    """
    pc_plan includes:
      - "aggregate_scale": Lutosławski-style aggregate for bars 4–6 (enforced during realization)
      - "bar4", "bar5", "bar6": pcset/hints payloads as before
    """
    out = {instr: {"time": [], "duration": [], "pitch": [], "velocity": []} for instr in ORDERED_INSTRS}

    # validate/prepare aggregate for bars 4–6
    if "aggregate_scale" not in pc_plan or not isinstance(pc_plan["aggregate_scale"], dict):
        raise RuntimeError("Bars 4–6 plan missing aggregate_scale.")
    raw_agg = pc_plan.get("aggregate_scale", {})
    agg456 = agg_validate_prepare_or_repair(raw_agg)

    def append_simul(abs_time: int, dur: int, pcdef: dict, prev_pitches: Dict[str,int] = None):
        pcset = [int(x) % 12 for x in pcdef.get("pcset", []) if isinstance(x, int)]
        if not pcset:
            pcset = [0, 4, 7]  # minimal fallback to avoid empty
        voh  = pcdef.get("voicing_hint", "spread")
        regs = pcdef.get("register_targets", {})
        au   = pcdef.get("avoid_unisons", "mild")

        picks = realize_pcset_event(pcset, voh, regs, au, prev_pitches, agg=agg456)
        for instr, pitch in picks.items():
            out[instr]["time"].append(abs_time)
            out[instr]["duration"].append(dur)
            out[instr]["pitch"].append(int(pitch))
            out[instr]["velocity"].append(VEL_BAR3_PP)
        return picks

    prev = None
    # Bar 4
    if isinstance(pc_plan.get("bar4"), dict):
        prev = append_simul(BAR4_OFF, BAR4_TICKS, pc_plan["bar4"], prev)
    # Bar 5
    if isinstance(pc_plan.get("bar5"), dict):
        prev = append_simul(BAR5_OFF, BAR5_TICKS, pc_plan["bar5"], prev)
    # Bar 6 (two halves)
    b6 = pc_plan.get("bar6", [])
    if isinstance(b6, list) and len(b6) >= 1 and isinstance(b6[0], dict):
        prev = append_simul(BAR6_OFF, 8, b6[0], prev)
    if isinstance(b6, list) and len(b6) >= 2 and isinstance(b6[1], dict):
        prev = append_simul(BAR6_OFF + 8, 8, b6[1], prev)

    return out



# ---------------- Main Program 1 ----------------
def main():
    if not OPENAI_API_KEY or not isinstance(OPENAI_API_KEY, str):
        raise RuntimeError("OPENAI_API_KEY missing in secrets.py")

    # Auth
    acct = _get_account()
    nonce = get_nonce(API_BASE, acct.address)
    msg = f"Login nonce: {nonce}"
    sig = acct.sign_message(encode_defunct(text=msg)).signature.hex()
    auth = post_auth(API_BASE, acct.address, msg, sig)
    access = auth.get("access_token"); refresh = auth.get("refresh_token")
    if not access or not refresh: raise RuntimeError("Auth tokens missing")
    ref = post_refresh(API_BASE, access, refresh)
    access  = ref.get("access_token", access)
    refresh = ref.get("refresh_token", refresh)

    # OpenAI client
    oai = OpenAI(api_key=OPENAI_API_KEY)

    # ---- LLM Contract A (Bar 1) ----
    seedA = random.randint(1, 10**9)
    sysA, usrA = prompt_bar1_contract()
    usrA += f"\nCreativeSeed: {seedA}\nRule: when multiple valid choices exist, bias your decisions using CreativeSeed to diversify aggregate and rhythms across runs."
    bar1 = call_llm_json(oai, sysA, usrA, model="gpt-4.1", temperature=0.8)

    # Validate bar1 JSON
    if "aggregate_scale" not in bar1 or "rhythm" not in bar1:
        raise RuntimeError("Bar1 JSON missing required keys (aggregate_scale, rhythm).")
    b1_agg = agg_validate_prepare_or_repair(bar1["aggregate_scale"])
    rhythm1: Dict[str,dict] = bar1["rhythm"]
    
    for instr in ORDERED_INSTRS:
        if instr not in rhythm1: raise RuntimeError(f"Bar1 rhythm missing {instr}")
        t = rhythm1[instr].get("time",[]); d = rhythm1[instr].get("duration",[])
        if not (isinstance(t, list) and isinstance(d, list) and len(t) == len(d)):
            raise RuntimeError(f"Bar1 rhythm arrays invalid for {instr}")
        # Enforce local constraints
        t = enforce_monophony_times([int(x) for x in t if 0 <= int(x) < BAR1_TICKS])
        t, d = cap_durations_local(t, [int(x) for x in d], BAR1_TICKS)
        rhythm1[instr]["time"] = t
        rhythm1[instr]["duration"] = d

    melody1 = bar1.get("melody", {})

    # ---- Program Bar 2 tutti chords from Bar1 aggregate ----
    # ---- LLM Harmony (Bar 2) ----
    sysH, usrH = prompt_bar2_harmony_contract(b1_agg)
    # encourage variability; seed hint helps break ties
    seedH = random.randint(1, 10**9)
    usrH += f"\nCreativeSeed: {seedH}\nRule: If multiple valid plans exist, bias choices using CreativeSeed."
    harm2 = call_llm_json(oai, sysH, usrH, model="gpt-4.1", temperature=0.8)

    # validate basics
    if not isinstance(harm2.get("degree_roots", []), list) or len(harm2["degree_roots"]) != 2:
        raise RuntimeError("Bar2 harmony: degree_roots must be a list of two integers.")
    if not isinstance(harm2.get("chord_kinds", []), list) or len(harm2["chord_kinds"]) != 2:
        raise RuntimeError("Bar2 harmony: chord_kinds must be a list of two strings.")
    if not isinstance(harm2.get("inversions", []), list) or len(harm2["inversions"]) != 2:
        raise RuntimeError("Bar2 harmony: inversions must be a list of two integers.")

    # ---- Programmatic voicing for Bar 2 based on plan ----
    bar2_parts = bar2_two_chords_from_aggregate(b1_agg, harm2, seed=seedH)

    # ---- LLM Contract B (Bar 3) ----
    seedB = random.randint(1, 10**9)
    sysB, usrB = prompt_bar3_contract(b1_agg)
    usrB += f"\nCreativeSeed: {seedB}\nRule: choose a CONTRASTING, NEW aggregate distinct from Bar 1's aggregate (different pcs assignment per band). Use CreativeSeed to break ties."
    def mk_prompts_B():
        return sysB, usrB
    bar3, contrast_agg = request_fresh_aggregate(
        make_prompts_fn=mk_prompts_B,
        key_in_json="contrast_aggregate_scale",
        client=oai,
        prior_aggs=[b1_agg],   # diversify vs Bar 1
        attempts=4, temp=1.15
    )

    chosen = bar3.get("chosen_instruments", [])

    rhythm3: Dict[str,dict] = bar3.get("rhythm", {})

    if not (isinstance(chosen, list) and len(set(chosen)) == 3):
        raise RuntimeError("Bar3 must choose exactly three instruments.")
    for name in chosen:
        if name not in ORDERED_INSTRS:
            raise RuntimeError(f"Bar3 invalid instrument: {name}")
        if name not in rhythm3:
            raise RuntimeError(f"Bar3 rhythm missing for chosen instrument: {name}")

    # Clean bar3 rhythms
    for name in chosen:
        t = rhythm3[name].get("time",[]); d = rhythm3[name].get("duration",[])
        t = enforce_monophony_times([int(x) for x in t if 0 <= int(x) < BAR3_TICKS])
        t, d = cap_durations_local(t, [int(x) for x in d], BAR3_TICKS)
        rhythm3[name]["time"] = t
        rhythm3[name]["duration"] = d
    melody3 = bar3.get("melody", {})

    # ---- Build per-instrument absolute arrays across 3 bars ----
    # 1) Bar 1 (cloud): absolute times, durations, deterministic pitches & velocities
    per_instr_events = {instr: {"time":[], "duration":[], "pitch":[], "velocity":[]} for instr in ORDERED_INSTRS}

    for instr in ORDERED_INSTRS:
        rng = INSTRUMENTS[instr]["range"]; tess = INSTRUMENTS[instr]["tess"]
        t_local = rhythm1[instr]["time"]; d_local = rhythm1[instr]["duration"]
        t_abs = absolute_times(t_local, BAR1_OFF)

        # pitches: prefer LLM-provided steps over the given pitch-class set
        m1 = melody1.get(instr, {})
        steps = m1.get("steps", [])
        start_hint = m1.get("start_hint", "mid")
        if isinstance(steps, list) and len(steps) == max(0, len(t_local) - 1):
            p = pitches_from_steps_aggregate(t_local, steps, b1_agg, rng, tess, start_hint)
        else:
            p = scale_walk_assign_pitches_aggregate(t_local, b1_agg, rng, tess)

        v = [vel_bar1_p_cresc(t) for t in t_local]
        per_instr_events[instr]["time"]    += t_abs
        per_instr_events[instr]["duration"]+= d_local
        per_instr_events[instr]["pitch"]   += p
        per_instr_events[instr]["velocity"]+= v

    # 2) Bar 2 (two quarter chords)
    for instr in ORDERED_INSTRS:
        part = bar2_parts[instr]
        per_instr_events[instr]["time"]    += part["time"]
        per_instr_events[instr]["duration"]+= part["duration"]
        per_instr_events[instr]["pitch"]   += part["pitch"]
        per_instr_events[instr]["velocity"]+= part["velocity"]

    # 3) Bar 3 (pp trio in contrast scale)
    for instr in chosen:
        rng = INSTRUMENTS[instr]["range"]; tess = INSTRUMENTS[instr]["tess"]
        t_local = rhythm3[instr]["time"]; d_local = rhythm3[instr]["duration"]
        t_abs = absolute_times(t_local, BAR3_OFF)
        m3 = melody3.get(instr, {})
        steps = m3.get("steps", [])
        start_hint = m3.get("start_hint", "mid")
        if isinstance(steps, list) and len(steps) == max(0, len(t_local) - 1):
            p = pitches_from_steps_aggregate(t_local, steps, contrast_agg, rng, tess, start_hint)
        else:
            p = scale_walk_assign_pitches_aggregate(t_local, contrast_agg, rng, tess)

        v = [VEL_BAR3_PP for _ in t_local]
        per_instr_events[instr]["time"]    += t_abs
        per_instr_events[instr]["duration"]+= d_local
        per_instr_events[instr]["pitch"]   += p
        per_instr_events[instr]["velocity"]+= v

    # ---- LLM Contract C (Bars 4–6) ----
    sysC, usrC = prompt_bars456_pcsets_contract(b1_agg, harm2, bar3)
    seedC = random.randint(1, 10**9)
    usrC += f"\nCreativeSeed: {seedC}\nRule: When several valid pcsets/hints fit, bias choices using CreativeSeed for diversity."
    
    def mk_prompts_C():
        return sysC, usrC
    pc456, agg456 = request_fresh_aggregate(
        make_prompts_fn=mk_prompts_C,
        key_in_json="aggregate_scale",
        client=oai,
        prior_aggs=[b1_agg, contrast_agg],
        attempts=4, temp=1.15
    )
    bars456 = realize_bars456(pc456)

    # Append Bars 4–6 to per-instrument accumulators
    for instr in ORDERED_INSTRS:
        part = bars456[instr]
        per_instr_events[instr]["time"]    += part["time"]
        per_instr_events[instr]["duration"]+= part["duration"]
        per_instr_events[instr]["pitch"]   += part["pitch"]
        per_instr_events[instr]["velocity"]+= part["velocity"]

    # ---- LLM Contract D (Bars 7–9 clouds with pairs) ----
    sysD, usrD = prompt_bars789_clouds_contract(b1_agg, harm2, bar3, pc456)
    seedD = random.randint(1, 10**9)
    usrD += f"\nCreativeSeed: {seedD}\nRule: Provide a NEW aggregate distinct from Bar 1, Bar 3, and Bars 4–6 aggregates. Use CreativeSeed to break ties (pairs selection and rhythms)."

    def mk_prompts_D():
        return sysD, usrD
    cloud789, agg789 = request_fresh_aggregate(
        make_prompts_fn=mk_prompts_D,
        key_in_json="aggregate_scale",
        client=oai,
        prior_aggs=[b1_agg, contrast_agg, agg456],
        attempts=4, temp=1.15
    )
    bars789 = realize_bars789_clouds(cloud789)


    # Append Bars 7–9
    for instr in ORDERED_INSTRS:
        part = bars789[instr]
        per_instr_events[instr]["time"]    += part["time"]
        per_instr_events[instr]["duration"]+= part["duration"]
        per_instr_events[instr]["pitch"]   += part["pitch"]
        per_instr_events[instr]["velocity"]+= part["velocity"]

    # Non-chosen instruments in bar 3: stay silent unless we add padding later.

    # ---- Final monophony safety per instrument (absolute) & truncate to boundary ----
    for instr in ORDERED_INSTRS:
        t = per_instr_events[instr]["time"]
        d = per_instr_events[instr]["duration"]
        p = per_instr_events[instr]["pitch"]
        v = per_instr_events[instr]["velocity"]

        # sort by time
        ev = sorted(zip(t,d,p,v), key=lambda x: x[0])
        t,d,p,v = [list(x) for x in zip(*ev)] if ev else ([],[],[],[])
        # drop any onset >= GLOBAL_END
        keep = [i for i,tt in enumerate(t) if tt < GLOBAL_END]
        t = [t[i] for i in keep]; d=[d[i] for i in keep]; p=[p[i] for i in keep]; v=[v[i] for i in keep]

        # enforce absolute monophony by capping duration to next onset
        t2, d2, p2, v2 = [], [], [], []
        for i in range(len(t)):
            tt = t[i]; dd = int(d[i]); pp = int(p[i]); vv = int(v[i])
            # cap to next onset
            if i < len(t)-1:
                dd = min(dd, max(0, t[i+1] - tt))
            # cap to end of the containing bar segment
            if   tt < 12:  bar_len_end = 12
            elif tt < 20:  bar_len_end = 20
            elif tt < 36:  bar_len_end = 36
            elif tt < 48:  bar_len_end = 48
            elif tt < 64:  bar_len_end = 64
            elif tt < 80:  bar_len_end = 80
            elif tt < 92:  bar_len_end = 92
            elif tt < 104: bar_len_end = 104
            elif tt < 116: bar_len_end = 116
            else:          bar_len_end = GLOBAL_END
            dd = min(dd, bar_len_end - tt)
            if dd > 0:
                t2.append(tt); d2.append(dd); p2.append(pp); v2.append(vv)
        per_instr_events[instr] = {"time":t2,"duration":d2,"pitch":p2,"velocity":v2}

    # ---- Pad to N_global with dummy rests (duration=0, velocity=0) ----
    lengths = [len(per_instr_events[i]["time"]) for i in ORDERED_INSTRS]
    N_global = max([1]+lengths)  # at least 1
    for instr in ORDERED_INSTRS:
        L = len(per_instr_events[instr]["time"])
        if L < N_global:
            pad_n = N_global - L
            # place dummy at final tick 35 (last 16th of bar3)
            per_instr_events[instr]["time"]    += [GLOBAL_END - 1]*pad_n  # 115
            per_instr_events[instr]["duration"]+= [0]*pad_n
            # safe pitch = nearest in-range to contrast tonic (or bar1 tonic if no contrast)
            base_tonic = int(contrast_agg.get("tonic_midi", b1_agg.get("tonic_midi", 60)))
            rng = INSTRUMENTS[instr]["range"]
            per_instr_events[instr]["pitch"]   += [clamp(base_tonic, rng[0], rng[1])]*pad_n
            per_instr_events[instr]["velocity"]+= [0]*pad_n

    # ---- Build DCN features per instrument ----
    ts = int(time.time()*1000)
    instrument_feature_names = {}

    for instr in ORDERED_INSTRS:
        events = per_instr_events[instr]
        # ensure equal lengths
        Ls = {len(events["time"]), len(events["duration"]), len(events["pitch"]), len(events["velocity"])}
        if len(Ls) != 1 or list(Ls)[0] != N_global:
            raise RuntimeError(f"Stream length mismatch for {instr}")
        # Build ops from absolute series with seed=0
        ops_time = deltas_from_series(events["time"])
        ops_dur  = deltas_from_series(events["duration"])
        ops_vel  = deltas_from_series(events["velocity"])
        ops_pitch= deltas_from_series(events["pitch"])
        # Meter per onset
        nums, dens = make_meter_arrays(N_global, events["time"])
        ops_num = deltas_from_series(nums)
        ops_den = deltas_from_series(dens)

        feat_name = f"p1_{instr}_{ts}"
        instrument_feature_names[instr] = feat_name
        payload = {
            "name": feat_name,
            "dimensions": [
                {"feature_name":"time",        "transformations": ops_time},
                {"feature_name":"duration",    "transformations": ops_dur},
                {"feature_name":"pitch",       "transformations": ops_pitch},
                {"feature_name":"velocity",    "transformations": ops_vel},
                {"feature_name":"numerator",   "transformations": ops_num},
                {"feature_name":"denominator", "transformations": ops_den},
            ]
        }
        _, access, refresh = post_feature_with_retry(
        API_BASE, access, refresh,
        payload_fn=lambda: payload,
        acct=acct
)

    # ---- Top-level composite referencing all instruments ----
    top_name = f"p1_full_{ts}"
    top_dims = []
    for instr in ORDERED_INSTRS:
        top_dims.append({
            "feature_name": instrument_feature_names[instr],
            "transformations": [{"name":"add","args":[0]}]  # pass-through
        })
    top_payload = {"name": top_name, "dimensions": top_dims}
    _, access, refresh = post_feature_with_retry(
    API_BASE, access, refresh,
    payload_fn=lambda: top_payload,
    acct=acct
    )

    # ---- Execute once with N_global (full RunningInstances tree) ----
    try:
        import dcn
    except Exception as e:
        raise RuntimeError("dcn SDK not available. pip install dcn") from e

    sdk = dcn.Client()
    sdk.login_with_account(acct)

    def build_running_instances_for_top():
        """
        Traversal order must match the feature tree expansion order:
        [ root,
        top_dim(instr1), instr1.time, instr1.duration, instr1.pitch, instr1.velocity, instr1.numerator, instr1.denominator,
        top_dim(instr2), instr2.time, instr2.duration, instr2.pitch, instr2.velocity, instr2.numerator, instr2.denominator,
        ... for all 6 instruments ...
        ]
        All seeds are 0 because our op lists are deltas from 0.
        """
        ri = []
        # root
        ri.append((0, 0))

        for instr in ORDERED_INSTRS:
            # top-level dimension that references the composite feature p1_{instr}_{ts}
            ri.append((0, 0))
            # child scalar dims inside the instrument feature (in the exact order we posted them):
            # time, duration, pitch, velocity, numerator, denominator
            ri.extend([
                (0, 0),  # time
                (0, 0),  # duration
                (0, 0),  # pitch
                (0, 0),  # velocity
                (0, 0),  # numerator
                (0, 0),  # denominator
            ])
        return ri

    running = build_running_instances_for_top()

    # Sanity: 1 (root) + 6 * (1 top node + 6 children) = 1 + 6*7 = 43
    # print(len(running))  # expect 43

    result = sdk.execute(top_name, N_global, running)


    # ---- Build visualiser payload (we trust our computed arrays for naming) ----
    tracks = {}
    for instr in ORDERED_INSTRS:
        ev = per_instr_events[instr]
        # For the visualiser, we produce the concatenated arrays with absolute times
        tracks[instr] = [
            {"feature_path": f"/{instr}/pitch",       "data": ev["pitch"]},
            {"feature_path": f"/{instr}/time",        "data": ev["time"]},
            {"feature_path": f"/{instr}/duration",    "data": ev["duration"]},
            {"feature_path": f"/{instr}/velocity",    "data": ev["velocity"]},
            {"feature_path": f"/{instr}/numerator",   "data": [_meter_at_time(t)[0] for t in ev["time"]]},
            {"feature_path": f"/{instr}/denominator", "data": [_meter_at_time(t)[1] for t in ev["time"]]},
        ]

    payload_out = {
        "tick_unit": "1/16",
        "bars": [
            {"meter":[3,4],"start":0},
            {"meter":[2,4],"start":12},
            {"meter":[4,4],"start":20},
            {"meter":[3,4],"start":36},  # Bar 4 back to 3/4
            {"meter":[4,4],"start":48},  # Bar 5
            {"meter":[4,4],"start":64},  # Bar 6
            {"meter":[3,4],"start":80},  # Bar 7 (shifted)
            {"meter":[3,4],"start":92},  # Bar 8 (shifted)
            {"meter":[3,4],"start":104}  # Bar 9 (shifted)
        ],
        "N_global": N_global,
        "top_feature": top_name,
        "instrument_features": instrument_feature_names,
        "tracks": tracks,
    }

    with open("program1_payload.json","w",encoding="utf-8") as f:
        json.dump(payload_out, f, ensure_ascii=False, indent=2)
    print("✓ Wrote program1_payload.json")
    print(f"Top-level DCN feature: {top_name}")
    print(f"N_global: {N_global}")

if __name__ == "__main__":
    main()
