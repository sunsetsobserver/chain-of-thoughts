#!/usr/bin/env python3

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
BAR6_TICKS = 16
BAR7_TICKS = 12  # 3/4
BAR8_TICKS = 12
BAR9_TICKS = 12
BAR10_TICKS = 12
BAR11_TICKS = 12
BAR12_TICKS = 12
BAR13_TICKS = 12
BAR14_TICKS = 12
BAR15_TICKS = 8    # 2/4
BAR16_TICKS = 16   # 4/4
BAR17_TICKS = 16   # 4/4
BAR18_TICKS = 16   # 4/4
BAR19_TICKS = 12   # ← 3/4 recap chord bar
BAR20_TICKS = 12   # 3/4
BAR21_TICKS = 12
BAR22_TICKS = 12
BAR23_TICKS = 12
BAR24_TICKS = 12
BAR25_TICKS = 12
BAR26_TICKS = 12
BAR27_TICKS = 12
BAR28_TICKS = 12
BAR29_TICKS = 12
BAR30_TICKS = 12

BAR1_OFF, BAR2_OFF, BAR3_OFF = 0, 12, 20
BAR4_OFF, BAR5_OFF, BAR6_OFF = 36, 48, 64
BAR7_OFF, BAR8_OFF, BAR9_OFF = 80, 92, 104
BAR10_OFF = 116
BAR11_OFF = 128
BAR12_OFF = 140
BAR13_OFF = 152
BAR14_OFF = 164
BAR15_OFF = 176
BAR16_OFF = 184
BAR17_OFF = 200
BAR18_OFF = 216
BAR19_OFF = 232               # 3/4 (12 ticks)
BAR20_OFF = 236               # 3/4
BAR21_OFF = 248
BAR22_OFF = 260
BAR23_OFF = 272
BAR24_OFF = 284
BAR25_OFF = 296
BAR26_OFF = 308
BAR27_OFF = 320
BAR28_OFF = 332
BAR29_OFF = 344
BAR30_OFF = 356

GLOBAL_END = 368

VEL_LOOP_BG = 48       # background loop dynamic
VEL_FORE_STAB = 92     # foreground staccato hits dynamic

VEL_JAZZ_CH1 = 86
VEL_JAZZ_CH2 = 92

VEL_BAR19_RECAP = 104     # mf–f recap hit
VEL_LONGTEXT_BASE = 84     # base for the long-note texture (± jitter)

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
    if t < 12:      return 3, 4   # bar 1
    elif t < 20:    return 2, 4   # bar 2
    elif t < 36:    return 4, 4   # bar 3
    elif t < 48:    return 3, 4   # bar 4
    elif t < 64:    return 4, 4   # bar 5
    elif t < 80:    return 4, 4   # bar 6
    elif t < 92:    return 3, 4   # bar 7
    elif t < 104:   return 3, 4   # bar 8
    elif t < 116:   return 3, 4   # bar 9
    elif t < 128:   return 3, 4   # bar 10
    elif t < 140:   return 3, 4   # bar 11
    elif t < 152:   return 3, 4   # bar 12
    elif t < 164:   return 3, 4   # bar 13
    elif t < 176:   return 3, 4   # bar 14
    elif t < 184:   return 2, 4   # bar 15
    elif t < 200:   return 4, 4   # bar 16
    elif t < 216:   return 4, 4   # bar 17
    elif t < 232:   return 4, 4   # bar 18
    elif t < 236:   return 1, 4   # bar 19
    elif t < 248:   return 3, 4   # bar 20
    elif t < 260:   return 3, 4   # bar 21
    elif t < 272:   return 3, 4   # bar 22
    elif t < 284:   return 3, 4   # bar 23
    elif t < 296:   return 3, 4   # bar 24
    elif t < 308:   return 3, 4   # bar 25
    elif t < 320:   return 3, 4   # bar 26
    elif t < 332:   return 3, 4   # bar 27
    elif t < 344:   return 3, 4   # bar 28
    elif t < 356:   return 3, 4   # bar 29
    elif t < 368:   return 3, 4   # bar 30
    else:           return 4, 4


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

def prompt_bar2_extra_aggregate(source_agg: dict, chord_pcs: List[int], tag: str) -> Tuple[str, str]:
    """
    Ask for a NEW Lutosławski-style aggregate to realize a specific Bar 2 chord.
    It must be valid (3–4 pcs per band, disjoint, union=12), non-equal-step, and materially
    different from Bar 1. We *hint* that the chord pcs should be distributed across ≥2 bands
    so voicings can stratify, but we still enforce the hard rules.
    """
    prev_map = pc_to_band_map(source_agg)
    chord_pcs_sorted = sorted(int(x) % 12 for x in chord_pcs)

    system = """
You output ONLY strict JSON. No prose.

Return:
{ "aggregate_scale": {
    "name":"string","tonic_midi":int,
    "bands":[{"midi_lo":int,"midi_hi":int,"pcs":[int,int,int(,int)]}, ...]
}}

Hard constraints:
- Each band's pcs length is 3 or 4.
- Bands’ pcs are pairwise disjoint; union across all bands is exactly 12.
- pcs are ABSOLUTE 0..11; ORDER within each band matters (used for degree stepping).
- Avoid pure equal-step bands (no [0,3,6,9] rotations; no [0,4,8]).

Validation: if any rule fails, regenerate internally and return only a valid JSON object.
"""
    user = (
        "Context (Bar 2 chord pcs to support): " + json.dumps(chord_pcs_sorted) + "\n"
        "Guidance (soft): distribute these pcs across at least two bands so verticals can stratify; "
        "set MIDI ranges sensibly (low/mid/high). Also produce material *meaningfully different* from Bar 1.\n"
        "Prev pc→band map (Bar 1) for diversity: " + json.dumps(prev_map) + "\n"
        f"Name hint: use something like \"bar2_{tag}\".\n"
        "Return VALID JSON ONLY."
    )
    return system.strip(), user.strip()

def request_bar2_aggregates(client: OpenAI, source_agg: dict, harm2: dict) -> Tuple[dict, dict]:
    """
    Build the two Bar 2 chord pc-sets from Bar 1's degree order,
    then request two fresh aggregates (A for chord1, B for chord2).
    Ensures diversity vs Bar 1 and vs the first new aggregate.
    """
    degree_order = agg_global_degree_order(source_agg)

    def chord_pcs_for(idx: int) -> List[int]:
        root = int(harm2["degree_roots"][idx]) % 12
        kind = str(harm2["chord_kinds"][idx])
        inv  = int(harm2["inversions"][idx])
        pcs  = _invert_pcset(_chord_pcset(degree_order, root, kind), inv)
        return [int(x) % 12 for x in pcs]

    chord1_pcs = chord_pcs_for(0)
    chord2_pcs = chord_pcs_for(1)

    # --- Aggregate A for chord 1 ---
    def mkA():
        return prompt_bar2_extra_aggregate(source_agg, chord1_pcs, "chordA")
    jsA, aggA = request_fresh_aggregate(
        make_prompts_fn=mkA,
        key_in_json="aggregate_scale",
        client=client,
        prior_aggs=[source_agg],   # diversify vs Bar 1
        attempts=4, temp=1.0
    )

    # --- Aggregate B for chord 2 (also diversify vs A) ---
    def mkB():
        return prompt_bar2_extra_aggregate(source_agg, chord2_pcs, "chordB")
    jsB, aggB = request_fresh_aggregate(
        make_prompts_fn=mkB,
        key_in_json="aggregate_scale",
        client=client,
        prior_aggs=[source_agg, aggA],
        attempts=4, temp=1.0
    )

    return aggA, aggB

def bar2_two_chords_with_extra_aggregates(source_agg: dict,
                                          aggA: dict, aggB: dict,
                                          plan: dict, seed: int = None) -> Dict[str, List[int]]:
    """
    Same timing/velocity as before (two quarter hits @ 12 and 16),
    but we *build the chord pcs* from Bar 1's degree order (source_agg),
    and *snap voicings* under per-chord aggregates (aggA for hit 1, aggB for hit 2).
    """
    rng = random.Random(seed)
    degree_order = agg_global_degree_order(source_agg)

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

    # keep your chair preferences as-is
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
        p1  = nearest_pitch_for_pc_aggregate(pc1, center_guess - widen, (lo, hi), aggA)

        pc2_pref = chord2_pcs_ord[i1 % len(chord2_pcs_ord)]
        p2  = nearest_pitch_for_pc_aggregate(pc2_pref, p1 + rng.choice([-2,-1,0,1,2]), (lo, hi), aggB)
        if p2 == p1:
            alt_pc2 = rng.choice(chord2_pcs_ord)
            p2 = nearest_pitch_for_pc_aggregate(alt_pc2, p1 + rng.choice([-3,-2,-1,1,2,3]), (lo, hi), aggB)

        voices[instr] = {
            "time": abs_times[:],
            "duration": durs[:],
            "pitch": [p1, p2],
            "velocity": [VEL_BAR2_FIRST, VEL_BAR2_SECOND],
        }
    return voices


def prompt_bar2_free_aggregate(tag: str, prior_map: dict) -> Tuple[str, str]:
    system = """
You output ONLY strict JSON. No prose.
Return:
{ "aggregate_scale": {
    "name":"string","tonic_midi":int,
    "bands":[{"midi_lo":int,"midi_hi":int,"pcs":[int,int,int(,int)]}, ...]
}}
Hard constraints:
- Each band's pcs length is 3 or 4.
- Bands’ pcs are pairwise disjoint; union across all bands is exactly 12.
- pcs are ABSOLUTE 0..11; ORDER within each band matters (used for degree stepping).
- Avoid pure equal-step bands (no [0,3,6,9] or [0,4,8] rotations).
Validation: if any rule fails, regenerate internally and return only a valid JSON object.
"""
    user = (
        f"Make a fresh, characterful aggregate for Bar 2 ({tag}). "
        "Aim for strong registral personality (sensible low/mid/high ranges), and be different from Bar 1.\n"
        "Prev pc→band map (Bar 1) for diversity: " + json.dumps(prior_map) + "\n"
        f"Name hint: 'bar2_{tag}'. Return VALID JSON ONLY."
    )
    return system.strip(), user.strip()


def request_bar2_two_free_aggregates(client: OpenAI, b1_agg: dict) -> Tuple[dict, dict]:
    prev_map = pc_to_band_map(b1_agg)

    def mkA():
        return prompt_bar2_free_aggregate("A", prev_map)
    jsA, aggA = request_fresh_aggregate(
        make_prompts_fn=mkA,
        key_in_json="aggregate_scale",
        client=client,
        prior_aggs=[b1_agg],   # diversify vs Bar 1
        attempts=4, temp=1.1
    )

    def mkB():
        return prompt_bar2_free_aggregate("B", prev_map)
    jsB, aggB = request_fresh_aggregate(
        make_prompts_fn=mkB,
        key_in_json="aggregate_scale",
        client=client,
        prior_aggs=[b1_agg, aggA],  # diversify vs Bar 1 and vs A
        attempts=4, temp=1.1
    )

    return aggA, aggB

def bar2_two_hits_from_free_aggs(aggA: dict, aggB: dict, seed: int = None) -> Dict[str, List[int]]:
    """
    Two quarter-note verticals @ absolute ticks 12 and 16 (dur=4 each).
    Hit 1 is voiced strictly under aggA, hit 2 strictly under aggB.
    For each instrument, pick a pc from the band covering its register and snap to nearest pitch.
    """
    rng = random.Random(seed)
    abs_times = [12, 16]
    durs = [4, 4]
    voices = {}

    for instr in ORDERED_INSTRS:
        lo, hi = INSTRUMENTS[instr]["range"]
        tess_lo, tess_hi = INSTRUMENTS[instr]["tess"]
        center = (tess_lo + tess_hi)//2

        # --- Hit 1 under aggA
        biA = agg_find_band_for_midi(center, aggA["bands"])
        bandA_pcs = aggA["bands"][biA]["pcs"]
        pc1 = bandA_pcs[rng.randrange(len(bandA_pcs))]
        p1  = nearest_pitch_for_pc_aggregate(pc1, center, (lo, hi), aggA)

        # --- Hit 2 under aggB (try a different pc if possible)
        biB = agg_find_band_for_midi(center, aggB["bands"])
        bandB_pcs = aggB["bands"][biB]["pcs"]
        choicesB = [pc for pc in bandB_pcs if pc != (p1 % 12)] or bandB_pcs[:]
        pc2 = choicesB[rng.randrange(len(choicesB))]
        # small offset so we don't always glue to the same octave
        p2  = nearest_pitch_for_pc_aggregate(pc2, p1 + rng.choice([-3,-2,-1,0,1,2,3]), (lo, hi), aggB)

        voices[instr] = {
            "time": abs_times[:],
            "duration": durs[:],
            "pitch": [int(p1), int(p2)],
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



def prompt_bars456_pcsets_contract(b1_agg: dict, bar3: dict) -> Tuple[str, str]:

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
    Prev pc→band maps to diverge from:
    - Bar1: {json.dumps(map1)}
    - Bar3: {json.dumps(map3)}
    Return VALID JSON ONLY.
    """.strip()
    return system.strip(), user.strip()

def prompt_bars789_clouds_contract(b1_agg: dict, bar3: dict, pc456: dict) -> Tuple[str, str]:
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

def prompt_bars1014_loop_stabs_contract(b1_agg: dict, bar3: dict, pc456: dict, cloud789: dict) -> Tuple[str,str]:
    system = """
    You output ONLY strict JSON. No prose.

    Task: Bars 10–14 (five bars of 3/4, 12 ticks each).
    - Pick EXACTLY three instruments to play a 1-bar background LOOP that repeats in all five bars.
    - The remaining three instruments play FOREGROUND staccato hits: short chords (and sometimes solo notes),
      rhythmically unexpected, always varying (no repeated chord per bar).

    Return:
    {
      "aggregate_scale": { "name":"string","tonic_midi":int,
        "bands":[{"midi_lo":int,"midi_hi":int,"pcs":[int,int,int(,int)]}, ...]
      },
      "background": {
        "loop_instruments": ["<instr>","<instr>","<instr>"],
        "loop_rhythm": {
          "<instr>": {"time":[ints 0..11 inc], "duration":[>0 ints, same len]},  // one bar only
          ...
        },
        "loop_melody": {
          "<instr>": {"start_hint":"low|mid|high", "steps":[len(time)-1 ints]},   // per chosen loop instrument
          ...
        }
      },
      "foreground": {
        "hit_instruments": ["<instr>","<instr>","<instr>"], // OPTIONAL; if omitted, it's the complement set
        "bars": [
          { "events": [
              {"time":int 0..11, "duration":int 1..3,
               "active":["subset of hit instruments (size 1..3)"],
               "pcset":[ints 0..11, len 2..6],
               "voicing_hint":"cluster|spread", "avoid_unisons":"mild|strong"
              },
              ...  // 3..8 events
          ]},
          ... // exactly 5 bar objects
        ]
      },
      "meta": { "bands_count": int, "band_lengths":[ints], "union_size": int }
    }

    Aggregate constraints (hard):
    - Each band's pcs length is 3 or 4; bands are disjoint; union across bands == 12; pcs are absolute (0..11).
    - Materially different pc→band mapping vs previous sections; avoid pure equal-step bands (prefer none).

    Background loop constraints:
    - loop_instruments are exactly 3 distinct names from the set.
    - For each loop instrument: monophony; times strictly increasing; time[i] + duration[i] <= 12.
    - steps length == len(time)-1.

    Foreground constraints:
    - 3..8 events per bar; durations 1..3; "active" is non-empty subset of hit_instruments (size 1..3).
    - Prefer off-beat placements overall; no more than ~1/3 events per bar at ticks {0,4,8}.
    - Chords should vary: within a bar, avoid duplicate unordered pcsets.

    Validation (must satisfy BEFORE returning JSON):
    - Aggregate passes hard rules; background arrays align; foreground is well-formed; bars count == 5.
    """

    map1 = pc_to_band_map(b1_agg)
    cagg = bar3.get("contrast_aggregate_scale", {})
    map3 = pc_to_band_map(cagg) if isinstance(cagg, dict) and "bands" in cagg else {}
    a456 = pc456.get("aggregate_scale", {})
    map456 = pc_to_band_map(a456) if isinstance(a456, dict) and "bands" in a456 else {}
    a789 = cloud789.get("aggregate_scale", {})
    map789 = pc_to_band_map(a789) if isinstance(a789, dict) and "bands" in a789 else {}

    user = (
        "Diversity vs previous (hard): at least 6 PCs must change band vs EACH prior section; also vary band ranges.\n"
        "Prev pc→band maps:\n"
        f"- Bar1: {json.dumps(map1)}\n"
        f"- Bar3: {json.dumps(map3)}\n"
        f"- Bars4–6: {json.dumps(map456)}\n"
        f"- Bars7–9: {json.dumps(map789)}\n"
        "RETURN RULES (hard):\n"
        "- background.loop_instruments: exactly 3 names.\n"
        "- For EACH loop instrument: loop_rhythm.time and .duration MUST be non-empty and valid (time[i]+duration[i] <= 12), "
        "and loop_melody.steps length MUST equal len(time)-1.\n"
        "- foreground.hit_instruments: exactly 3 names (explicit, not implied).\n"
        "- foreground.bars: exactly 5 bars; each bar has 3..8 events; each event has non-empty 'active' (subset of hit instruments), "
        "'pcset' (len 2..6, ints 0..11), time 0..11, duration 1..3, and voicing/avoid fields.\n"
        "- Empty arrays or missing fields are INVALID; regenerate internally and return a valid JSON object.\n"
        "Return VALID JSON ONLY."
    )

    return system.strip(), user.strip()

def _err(errors, msg): errors.append(msg)

def validate_bars1014_plan_or_raise(plan: dict) -> dict:
    """Validate Contract E plan. Raise with a precise list of problems; return the validated aggregate."""
    errors = []

    # --- Aggregate ---
    agg_raw = plan.get("aggregate_scale")
    if not isinstance(agg_raw, dict):
        _err(errors, "aggregate_scale missing or not an object.")
    else:
        try:
            agg_validate_prepare_or_repair(agg_raw)
        except Exception as e:
            _err(errors, f"aggregate_scale invalid: {e}")

    # --- Background (exactly 3 loop instruments, with non-empty rhythms and proper steps) ---
    bg = plan.get("background")
    if not isinstance(bg, dict):
        _err(errors, "background missing or not an object.")
        bg = {}
    loop_instruments = bg.get("loop_instruments")
    if not (isinstance(loop_instruments, list) and len(set(loop_instruments)) == 3 and
            all(i in ORDERED_INSTRS for i in loop_instruments)):
        _err(errors, "background.loop_instruments must list exactly 3 valid instruments (unique).")

    loop_rhythm = bg.get("loop_rhythm", {})
    loop_melody = bg.get("loop_melody", {})
    if isinstance(loop_instruments, list):
        for instr in loop_instruments:
            r = (loop_rhythm or {}).get(instr)
            if not (isinstance(r, dict) and isinstance(r.get("time"), list) and isinstance(r.get("duration"), list)):
                _err(errors, f"loop_rhythm missing arrays for {instr}.")
                continue
            t = [int(x) for x in r["time"]]
            d = [int(x) for x in r["duration"]]
            if len(t) == 0 or len(t) != len(d):
                _err(errors, f"loop_rhythm arrays invalid for {instr} (empty or len mismatch).")
            if any(x < 0 or x > 11 for x in t):
                _err(errors, f"loop_rhythm.time out of 0..11 for {instr}.")
            if any(dd <= 0 for dd in d):
                _err(errors, f"loop_rhythm.duration must be positive for {instr}.")
            # bar cap
            for ti, di in zip(t, d):
                if ti + di > 12: _err(errors, f"loop note exceeds bar (t+dur>12) for {instr}.")
            # melody checks
            m = (loop_melody or {}).get(instr)
            if not (isinstance(m, dict) and isinstance(m.get("steps"), list)):
                _err(errors, f"loop_melody.steps missing for {instr}.")
            else:
                if len(m["steps"]) != max(0, len(t) - 1):
                    _err(errors, f"loop_melody.steps length must be len(time)-1 for {instr}.")

    # --- Foreground (explicit 3 names; 5 bars; 3..8 events each; each event complete) ---
    fg = plan.get("foreground")
    if not isinstance(fg, dict):
        _err(errors, "foreground missing or not an object.")
        fg = {}
    hit_instruments = fg.get("hit_instruments")
    if not (isinstance(hit_instruments, list) and len(set(hit_instruments)) == 3 and
            all(i in ORDERED_INSTRS for i in hit_instruments)):
        _err(errors, "foreground.hit_instruments must list exactly 3 valid instruments (explicit).")

    # disjointness check (hits vs loop)
    if isinstance(hit_instruments, list) and isinstance(loop_instruments, list):
        if set(hit_instruments) & set(loop_instruments):
            _err(errors, "hit_instruments must be disjoint from loop_instruments.")

    bars = fg.get("bars")
    if not (isinstance(bars, list) and len(bars) == 5):
        _err(errors, "foreground.bars must be a list of exactly 5 bar objects.")
        bars = []

    for i, bar in enumerate(bars):
        evs = (bar or {}).get("events")
        if not (isinstance(evs, list) and 3 <= len(evs) <= 8):
            _err(errors, f"bar {10+i}: events must be 3..8.")
            continue
        for j, ev in enumerate(evs):
            if not isinstance(ev, dict):
                _err(errors, f"bar {10+i} event {j}: not an object."); continue
            t = ev.get("time"); dur = ev.get("duration")
            active = ev.get("active"); pcset = ev.get("pcset")
            voh = ev.get("voicing_hint"); au = ev.get("avoid_unisons")
            if not (isinstance(t, int) and 0 <= t <= 11): _err(errors, f"bar {10+i} event {j}: time must be 0..11.")
            if not (isinstance(dur, int) and 1 <= dur <= 3): _err(errors, f"bar {10+i} event {j}: duration must be 1..3.")
            if not (isinstance(active, list) and 1 <= len(active) <= 3 and
                    all(a in (hit_instruments or []) for a in active)):
                _err(errors, f"bar {10+i} event {j}: active must be non-empty subset of hit_instruments.")
            if not (isinstance(pcset, list) and 2 <= len(pcset) <= 6 and all(isinstance(x, int) and 0 <= x <= 11 for x in pcset)):
                _err(errors, f"bar {10+i} event {j}: pcset must be ints 0..11 (len 2..6).")
            if voh not in ("cluster","spread"): _err(errors, f"bar {10+i} event {j}: voicing_hint must be 'cluster' or 'spread'.")
            if au not in ("mild","strong"): _err(errors, f"bar {10+i} event {j}: avoid_unisons must be 'mild' or 'strong'.")

    if errors:
        bullets = "\n".join(f"- {e}" for e in errors)
        raise RuntimeError(f"Bars 10–14 plan invalid:\n{bullets}")

    # return validated aggregate (prepared)
    return agg_validate_prepare_or_repair(agg_raw)

def request_bars1014_strict(client: OpenAI, b1_agg: dict, bar3: dict, pc456: dict, cloud789: dict,
                            attempts: int = 4, temp: float = 1.1) -> Tuple[dict, dict]:
    """
    Ask for Contract E until it passes strict validation, else raise with the last error list.
    """
    last_err = None

    SCHEMA_TEMPLATE = """
    {
    "aggregate_scale": {
        "name": "string",
        "tonic_midi": 60,
        "bands": [
        {"midi_lo": 28, "midi_hi": 47, "pcs": [0,0,0(,0)]},
        {"midi_lo": 48, "midi_hi": 71, "pcs": [0,0,0(,0)]},
        {"midi_lo": 72, "midi_hi": 103, "pcs": [0,0,0(,0)]}
        ]
    },
    "background": {
        "loop_instruments": ["<exactly-3>"],
        "loop_rhythm": {
        "<instrA>": {"time":[...0..11...], "duration":[...]},
        "<instrB>": {"time":[...], "duration":[...]},
        "<instrC>": {"time":[...], "duration":[...]}
        },
        "loop_melody": {
        "<instrA>": {"start_hint":"low|mid|high", "steps":[...]},
        "<instrB>": {"start_hint":"low|mid|high", "steps":[...]},
        "<instrC>": {"start_hint":"low|mid|high", "steps":[...]}
        }
    },
    "foreground": {
        "hit_instruments": ["<exactly-3>"],
        "bars": [
        {"events":[
            {"time":0, "duration":1, "active":["<subset-of-hit>"], "pcset":[0,0], "voicing_hint":"cluster", "avoid_unisons":"strong"}
        ]},
        {"events":[...]},
        {"events":[...]},
        {"events":[...]},
        {"events":[...]}
        ]
    },
    "meta": {"bands_count": 3, "band_lengths": [3,4,5], "union_size": 12}
    }
    """.strip()

    for k in range(max(1, attempts)):
        sysE, usrE = prompt_bars1014_loop_stabs_contract(b1_agg, bar3, pc456, cloud789)
        usrE += (
            "\nINSTRUMENT UNIVERSE (use only these names): "
            "['alto_flute','violin','bass_clarinet','trumpet','cello','double_bass']\n"
            "Pick EXACTLY 3 for background.loop_instruments; "
            "foreground.hit_instruments must be EXACTLY the other 3 and listed explicitly.\n"
            "Use this exact JSON shape (no extra keys, no comments). Fill all placeholders with concrete values:\n"
            + SCHEMA_TEMPLATE
        )
        plan = call_llm_json(client, sysE, usrE, model="gpt-4.1", temperature=0.6, top_p=0.9)
        try:
            agg = validate_bars1014_plan_or_raise(plan)
            return plan, agg
        except RuntimeError as e:
            # keep only the bullet lines for compact feedback
            last_err = "\n".join(line for line in str(e).splitlines() if line.startswith("- "))
            continue
    raise RuntimeError("Contract E failed validation after retries:\n" + (last_err or "(no details)"))


def realize_bars1014(plan: dict, validated_agg: dict = None) -> Dict[str, Dict[str, List[int]]]:
    """
    Realize Bars 10–14 from a validated Contract E plan.
    Assumes plan already passed validate_bars1014_plan_or_raise (no defaults here).
    """
    out = {instr: {"time": [], "duration": [], "pitch": [], "velocity": []} for instr in ORDERED_INSTRS}

    # Use the validated aggregate if provided; otherwise repair/validate the plan's aggregate.
    agg = validated_agg if isinstance(validated_agg, dict) else agg_validate_prepare_or_repair(plan.get("aggregate_scale", {}))

    # Background
    bg = plan["background"]
    loop_instruments = list(bg["loop_instruments"])
    loop_rhythm = bg["loop_rhythm"]; loop_melody = bg["loop_melody"]

    loop_cache = {}
    for instr in loop_instruments:
        L = 12
        r = loop_rhythm[instr]
        t_local = enforce_monophony_times([int(x) for x in r["time"] if 0 <= int(x) < L])
        d_local = [int(x) for x in r["duration"]]
        t_local, d_local = cap_durations_local(t_local, d_local, L)

        m = loop_melody[instr]
        steps = m["steps"]
        start_hint = m.get("start_hint","mid")
        rng = INSTRUMENTS[instr]["range"]; tess = INSTRUMENTS[instr]["tess"]
        pitches = (pitches_from_steps_aggregate(t_local, steps, agg, rng, tess, start_hint)
                   if len(steps) == max(0, len(t_local)-1)
                   else scale_walk_assign_pitches_aggregate(t_local, agg, rng, tess))
        loop_cache[instr] = (t_local, d_local, pitches)

    bar_offsets = [BAR10_OFF, BAR11_OFF, BAR12_OFF, BAR13_OFF, BAR14_OFF]
    for off in bar_offsets:
        for instr in loop_instruments:
            t_local, d_local, pitches = loop_cache[instr]
            out[instr]["time"]    += absolute_times(t_local, off)
            out[instr]["duration"]+= d_local
            out[instr]["pitch"]   += pitches
            out[instr]["velocity"]+= [VEL_LOOP_BG]*len(t_local)

    # Foreground
    fg = plan["foreground"]
    hit_instruments = list(fg["hit_instruments"])
    bars = fg["bars"]

    prev_pitches = {}
    for idx, bar in enumerate(bars):
        off = bar_offsets[idx]
        events = sorted(bar["events"], key=lambda e: int(e["time"]))
        for ev in events:
            t = int(ev["time"]); dur = int(ev["duration"])
            active = [i for i in ev["active"] if i in hit_instruments]
            pcset = [int(x) % 12 for x in ev["pcset"]]
            voh = ev.get("voicing_hint","cluster")
            au  = ev.get("avoid_unisons","strong")
            regs = ev.get("register_targets", {}) or {}

            picks = realize_pcset_event_subset(pcset, voh, regs, au, active, prev_pitches, agg=agg)
            for instr, pitch in picks.items():
                out[instr]["time"].append(off + t)
                out[instr]["duration"].append(dur)
                out[instr]["pitch"].append(int(pitch))
                out[instr]["velocity"].append(VEL_FORE_STAB)
                prev_pitches[instr] = int(pitch)

    return out

def realize_longnote_texture_20_30(jazz_aggs: List[dict], seed: int = None) -> Dict[str, Dict[str, List[int]]]:
    """
    Bars 20–30 (all 3/4): one continuous long-note texture, generated in a single pass
    to avoid any 25→26 seam. We still color-shift using three jazzy aggregates:
      [20–23) = A, [23–26) = B, [26–30] = C   (half-bar overlap windows are okay; notes may cross boundaries)
    """
    rng = random.Random(seed)
    aggA = jazz_aggs[0]
    aggB = jazz_aggs[1 if len(jazz_aggs) > 1 else 0]
    aggC = jazz_aggs[2 if len(jazz_aggs) > 2 else (1 if len(jazz_aggs) > 1 else 0)]

    region_lo, region_hi = BAR20_OFF, BAR30_OFF + BAR30_TICKS  # 236..368 per your constants

    def agg_for_time(tt: int):
        if tt < BAR24_OFF: return aggA
        if tt < BAR26_OFF: return aggB
        return aggC

    out = {instr: {"time": [], "duration": [], "pitch": [], "velocity": []} for instr in ORDERED_INSTRS}

    # Slightly higher counts than the 20–25 version, since we cover 11 bars now (20..30 inclusive)
    base_counts = {
        "double_bass":   5,
        "cello":         5,
        "bass_clarinet": 6,
        "trumpet":       5,
        "violin":        6,
        "alto_flute":    6,
    }

    # Chair-based center offsets (kept from your voicing logic)
    centers = {
        "double_bass": -12, "cello": -7, "bass_clarinet": -3,
        "trumpet": +2, "violin": +7, "alto_flute": +9,
    }

    for instr in ORDERED_INSTRS:
        lo, hi = INSTRUMENTS[instr]["range"]
        tess_lo, tess_hi = INSTRUMENTS[instr]["tess"]
        center_guess = (tess_lo + tess_hi)//2 + centers.get(instr, 0)

        n = max(1, base_counts[instr] + rng.choice([-1,0,0,1]))
        # sample onsets across the WHOLE region (single pass → no seam)
        starts = []
        for _ in range(n * 2):  # oversample, then trim to n unique
            bar_start = rng.choice([BAR20_OFF, BAR21_OFF, BAR22_OFF, BAR23_OFF, BAR24_OFF, BAR25_OFF,
                                    BAR26_OFF, BAR27_OFF, BAR28_OFF, BAR29_OFF, BAR30_OFF])
            # prefer early-in-bar entries but allow variety
            inbar = rng.choice([0,1,2,3,4,5,6,7,8,9])
            starts.append(bar_start + inbar)
        starts = sorted(sorted(set(starts))[:n])

        prev_pc = None
        for st in starts:
            if st >= region_hi:
                continue
            dur = rng.choice([8,10,12,14,16,18])  # long sustains
            end = min(region_hi, st + dur)
            dur = max(1, end - st)

            agg = agg_for_time(st)
            bi  = agg_find_band_for_midi(center_guess, agg["bands"])
            pcs = list(agg["bands"][bi]["pcs"])
            pc  = pcs[rng.randrange(len(pcs))] if prev_pc is None else _pick_pc_close_to(prev_pc, pcs)

            pitch = nearest_pitch_for_pc_aggregate(int(pc),
                                                   center_guess + rng.choice([-2,-1,0,1,2]),
                                                   (lo, hi), agg)
            vel = clamp(VEL_LONGTEXT_BASE + rng.choice([-8,-6,-4,-2,0,0,0,2,4,6,8]), 40, 110)

            out[instr]["time"].append(int(st))
            out[instr]["duration"].append(int(dur))
            out[instr]["pitch"].append(int(pitch))
            out[instr]["velocity"].append(vel)
            prev_pc = int(pc)

    return out

def prompt_bars1618_loop_stabs_contract(b1_agg: dict, bar3: dict, pc456: dict, cloud789: dict, plan1014: dict) -> Tuple[str, str]:
    """
    Contract F: Bars 16–18, same principle as Bars 10–14 but in 4/4.
    We provide a concise summary of Bars 10–14 so the model can continue intelligently.
    """
    # --- Build a small context summary from Bars 10–14 plan ---
    bg = (plan1014 or {}).get("background", {}) or {}
    fg = (plan1014 or {}).get("foreground", {}) or {}
    prev_loop = bg.get("loop_instruments", [])
    prev_hit  = fg.get("hit_instruments", [])
    # summarize loop rhythms (times only to keep prompt compact)
    prev_loop_times = {}
    for i in prev_loop:
        r = (bg.get("loop_rhythm", {}) or {}).get(i, {})
        prev_loop_times[i] = (r.get("time") or [])[:]

    # summarize foreground pcsets used per bar (unique unordered pcsets)
    prev_fg_summary = []
    for bar in (fg.get("bars") or []):
        pcs = []
        for ev in (bar.get("events") or []):
            if isinstance(ev.get("pcset"), list):
                pcs.append(sorted([int(x) % 12 for x in ev["pcset"]]))
        # dedup
        uniq = []
        seen = set()
        for s in pcs:
            tpl = tuple(s)
            if tpl not in seen:
                uniq.append(s); seen.add(tpl)
        prev_fg_summary.append({"unique_pcsets": uniq})

    context = {
        "bars_10_14": {
            "loop_instruments": prev_loop,
            "loop_times": prev_loop_times,
            "hit_instruments": prev_hit,
            "foreground_summary": prev_fg_summary
        }
    }

    # pc→band maps (helpful but keep it light)
    map1   = pc_to_band_map(b1_agg)
    map3   = pc_to_band_map(bar3.get("contrast_aggregate_scale", {})) if isinstance(bar3.get("contrast_aggregate_scale", {}), dict) else {}
    map456 = pc_to_band_map(pc456.get("aggregate_scale", {})) if isinstance(pc456.get("aggregate_scale", {}), dict) else {}
    map789 = pc_to_band_map(cloud789.get("aggregate_scale", {})) if isinstance(cloud789.get("aggregate_scale", {}), dict) else {}
    map1014= pc_to_band_map(plan1014.get("aggregate_scale", {})) if isinstance(plan1014.get("aggregate_scale", {}), dict) else {}

    system = """
    You output ONLY strict JSON. No prose.

    Task: Bars 16–18 (THREE bars of 4/4, 16 ticks each).
    Principle identical to Bars 10–14:
      - Pick EXACTLY three instruments to play a 1-bar background LOOP that repeats in all three bars.
      - The other three instruments play FOREGROUND staccato hits: short chords (and sometimes solo notes),
        rhythmically unexpected, always varying (no repeated chord per bar).

    Return:
    {
      "aggregate_scale": { "name":"string","tonic_midi":int,
        "bands":[{"midi_lo":int,"midi_hi":int,"pcs":[int,int,int(,int)]}, ...]
      },
      "background": {
        "loop_instruments": ["<instr>","<instr>","<instr>"],
        "loop_rhythm": {
          "<instr>": {"time":[ints 0..15 inc], "duration":[>0 ints, same len]},  // one bar only
          ...
        },
        "loop_melody": {
          "<instr>": {"start_hint":"low|mid|high", "steps":[len(time)-1 ints]},
          ...
        }
      },
      "foreground": {
        "hit_instruments": ["<instr>","<instr>","<instr>"], // must be the complement set
        "bars": [
          { "events": [
              {"time":int 0..15, "duration":int 1..3,
               "active":["subset of hit instruments (size 1..3)"],
               "pcset":[ints 0..11, len 2..6],
               "voicing_hint":"cluster|spread", "avoid_unisons":"mild|strong"
              },
              ...  // 3..8 events
          ]},
          { ... }, { ... }  // exactly 3 bar objects
        ]
      },
      "meta": { "bands_count": int, "band_lengths":[ints], "union_size": int }
    }

    Aggregate constraints (hard):
    - Each band's pcs length is 3 or 4; bands are disjoint; union across bands == 12; pcs are absolute (0..11).
    - Avoid pure equal-step bands (prefer none).

    Background constraints:
    - loop_instruments are exactly 3 distinct names from the set below.
    - For each loop instrument: monophony; times strictly increasing; time[i] + duration[i] <= 16.
    - steps length == len(time)-1.

    Foreground constraints:
    - 3..8 events per bar; durations 1..3; "active" is non-empty subset of hit_instruments (size 1..3).
    - Prefer off-beat placements overall; no more than ~1/3 events per bar at ticks {0,4,8,12}.
    - Within a bar, avoid duplicate unordered pcsets.

    Validation (must satisfy BEFORE returning JSON):
    - Aggregate passes hard rules; background arrays align; foreground is well-formed; bars count == 3.
    - hit_instruments must be exactly the complement of loop_instruments (disjoint, union is all six).
    """

    user = (
      "INSTRUMENT UNIVERSE (use only these names): "
      "['alto_flute','violin','bass_clarinet','trumpet','cello','double_bass']\n"
      "CONTEXT from Bars 10–14 (use to CONTINUE intelligently, not to copy):\n"
      + json.dumps(context)
      + "\nPrev pc→band maps (for reference only):\n"
      f"- Bar1: {json.dumps(map1)}\n"
      f"- Bar3: {json.dumps(map3)}\n"
      f"- Bars4–6: {json.dumps(map456)}\n"
      f"- Bars7–9: {json.dumps(map789)}\n"
      f"- Bars10–14: {json.dumps(map1014)}\n"
      "Return VALID JSON ONLY."
    )

    return system.strip(), user.strip()

def prompt_jazzy_aggregate(tag: str, prior_maps: List[Dict[int,int]]) -> Tuple[str,str]:
    """
    Ask for a NEW Lutosławski-style aggregate with a 'jazzy' color:
    - within-band pcs should afford many minor/major 3rds and perfect 4ths between neighbors,
      but still obey: 3–4 pcs per band, bands disjoint, union of pcs == 12, order matters.
    - Avoid pure equal-step cycles; make it contrast with prior material.
    """
    system = """
You output ONLY strict JSON. No prose.
Return:
{ "aggregate_scale": {
    "name":"string","tonic_midi":int,
    "bands":[{"midi_lo":int,"midi_hi":int,"pcs":[int,int,int(,int)]}, ...]
}}
Hard constraints (validate before returning):
- Each band pcs length is 3 or 4.
- Bands’ pcs are pairwise disjoint; union across bands is exactly 12.
- pcs are ABSOLUTE 0..11; ORDER within each band matters (used for degree stepping).
- Avoid pure equal-step bands (no [0,3,6,9] or [0,4,8] rotations).
"""
    user = (
        f"Make a fresh, contrasting aggregate for a sustained CHORD ({tag}) with a subtle 'jazzy' color.\n"
        "- Preference: within each band, ordered pcs should afford many m3/M3/P4 relationships between neighbors.\n"
        "- Set MIDI ranges sensibly (low/mid/high) to match orchestral registers.\n"
        "Diversity guidance (prev pc→band maps): " + json.dumps(prior_maps) + "\n"
        f"Name hint: 'jazzy_{tag}'. Return VALID JSON ONLY."
    )
    return system.strip(), user.strip()

def request_two_jazzy_aggregates(client: OpenAI, prior_aggs: List[dict]) -> Tuple[dict, dict]:
    prior_maps = [pc_to_band_map(a) for a in prior_aggs if isinstance(a, dict) and a.get("bands")]
    def mkA(): return prompt_jazzy_aggregate("A", prior_maps)
    jsA, aggA = request_fresh_aggregate(
        make_prompts_fn=mkA, key_in_json="aggregate_scale",
        client=client, prior_aggs=prior_aggs, attempts=5, temp=1.15
    )
    def mkB(): return prompt_jazzy_aggregate("B", prior_maps)
    jsB, aggB = request_fresh_aggregate(
        make_prompts_fn=mkB, key_in_json="aggregate_scale",
        client=client, prior_aggs=prior_aggs+[aggA], attempts=5, temp=1.15
    )
    return aggA, aggB

def _pick_pc_close_to(target_pc: int, candidates: List[int]) -> int:
    """Choose candidate closest (circularly) to target_pc or to target_pc±3/±4 as jazz-biased anchors."""
    target_pc = int(target_pc) % 12
    anchors = [(target_pc + k) % 12 for k in (0, +3, -3, +4, -4)]
    def circ(a,b): 
        d = abs((a-b) % 12)
        return min(d, 12 - d)
    return min(candidates, key=lambda c: min(circ(c, a) for a in anchors))

def realize_twochord_sixbar_jazzy(agg1: dict, agg2: dict, seed: int = None) -> Dict[str, Dict[str, List[int]]]:
    """
    Bars 20–25 (each 12 ticks): Chord 1 staggered over bars 20–21; Chord 2 tutti at bar 22 beat 1;
    releases across bar 23 (beats 1–3) and bar 24 (beats 1–2); bar 25 = Alto Flute solo sustain.
    """
    rng = random.Random(seed)
    out = {instr: {"time": [], "duration": [], "pitch": [], "velocity": []} for instr in ORDERED_INSTRS}

    # --- chord 1 entrances (absolute times + durations up to bar 22 beat 1) ---
    ch1_entries = {
        "double_bass":  BAR20_OFF + 0,   # bar 20 beat 1
        "cello":        BAR20_OFF + 4,   # bar 20 beat 2
        "bass_clarinet":BAR20_OFF + 8,   # bar 20 beat 3
        "trumpet":      BAR21_OFF + 0,   # bar 21 beat 1
        "violin":       BAR21_OFF + 4,   # bar 21 beat 2
        "alto_flute":   BAR21_OFF + 8,   # bar 21 beat 3
    }
    ch1_end = BAR22_OFF  # all hold until bar 22 beat 1
    # --- chord 2 global start ---
    ch2_start = BAR22_OFF

    # release order & times (one per beat starting bar 23 beat 1)
    # Using the order you asked for, with CELLO inserted 2nd so all six are covered; Alto Flute is excluded to leave it solo.
    releases = [
        ("double_bass",   BAR23_OFF + 0),  # bar 23 beat 1
        ("cello",         BAR23_OFF + 4),  # bar 23 beat 2
        ("bass_clarinet", BAR23_OFF + 8),  # bar 23 beat 3
        ("trumpet",       BAR24_OFF + 0),  # bar 24 beat 1
        ("violin",        BAR24_OFF + 4),  # bar 24 beat 2
    ]
    af_end = BAR25_OFF + 12  # Alto Flute alone for whole bar 25

    # --- choose chord pcs & pitches ---
    for instr in ORDERED_INSTRS:
        lo, hi = INSTRUMENTS[instr]["range"]
        tess_lo, tess_hi = INSTRUMENTS[instr]["tess"]
        center = (tess_lo + tess_hi)//2

        # chord 1 pc: pick from the band covering the center
        bi1 = agg_find_band_for_midi(center, agg1["bands"])
        pcs1 = list(agg1["bands"][bi1]["pcs"])
        pc1  = pcs1[rng.randrange(len(pcs1))]
        p1   = nearest_pitch_for_pc_aggregate(pc1, center, (lo,hi), agg1)

        # chord 2 pc: from agg2 band at center, biased close to pc1 ±3/±4
        bi2 = agg_find_band_for_midi(center, agg2["bands"])
        pcs2 = list(agg2["bands"][bi2]["pcs"])
        pc2  = _pick_pc_close_to(pc1, pcs2)
        p2   = nearest_pitch_for_pc_aggregate(pc2, p1 + rng.choice([-2,-1,0,1,2]), (lo,hi), agg2)

        # --- write chord 1 event if the instrument has an entry ---
        t1 = ch1_entries.get(instr, None)
        if t1 is not None:
            d1 = max(1, ch1_end - t1)
            out[instr]["time"].append(t1)
            out[instr]["duration"].append(d1)
            out[instr]["pitch"].append(int(p1))
            out[instr]["velocity"].append(VEL_JAZZ_CH1)

        # --- chord 2 event for everyone, with instrument-specific release ---
        # default end is af_end for alto_flute, else until its scheduled release or end of bar 22 only for DB (if first release).
        if instr == "alto_flute":
            t2_end = af_end  # solo through bar 25
        else:
            rel_map = {name: t for name, t in releases}
            t2_end = rel_map.get(instr, ch2_start + 12)  # if not listed, at least sustain bar 22

        d2 = max(1, t2_end - ch2_start)
        out[instr]["time"].append(ch2_start)
        out[instr]["duration"].append(d2)
        out[instr]["pitch"].append(int(p2))
        out[instr]["velocity"].append(VEL_JAZZ_CH2)

    return out



def validate_bars1618_plan_or_raise(plan: dict) -> dict:
    errors = []

    # Aggregate
    agg_raw = plan.get("aggregate_scale")
    if not isinstance(agg_raw, dict):
        errors.append("aggregate_scale missing or not an object.")
    else:
        try:
            agg_validate_prepare_or_repair(agg_raw)
        except Exception as e:
            errors.append(f"aggregate_scale invalid: {e}")

    # Background
    bg = plan.get("background")
    if not isinstance(bg, dict):
        errors.append("background missing or not an object.")
        bg = {}
    loop_instruments = bg.get("loop_instruments")
    if not (isinstance(loop_instruments, list) and len(set(loop_instruments)) == 3 and
            all(i in ORDERED_INSTRS for i in loop_instruments)):
        errors.append("background.loop_instruments must list exactly 3 valid instruments (unique).")

    loop_rhythm = bg.get("loop_rhythm", {}) or {}
    loop_melody = bg.get("loop_melody", {}) or {}
    if isinstance(loop_instruments, list):
        for instr in loop_instruments:
            r = loop_rhythm.get(instr)
            if not (isinstance(r, dict) and isinstance(r.get("time"), list) and isinstance(r.get("duration"), list)):
                errors.append(f"loop_rhythm missing arrays for {instr}."); continue
            t = [int(x) for x in r["time"]]
            d = [int(x) for x in r["duration"]]
            if len(t) == 0 or len(t) != len(d):
                errors.append(f"loop_rhythm arrays invalid for {instr} (empty or len mismatch).")
            if any(x < 0 or x > 15 for x in t):
                errors.append(f"loop_rhythm.time out of 0..15 for {instr}.")
            if any(dd <= 0 for dd in d):
                errors.append(f"loop_rhythm.duration must be positive for {instr}.")
            for ti, di in zip(t, d):
                if ti + di > 16: errors.append(f"loop note exceeds bar (t+dur>16) for {instr}.")
            m = loop_melody.get(instr)
            if not (isinstance(m, dict) and isinstance(m.get("steps"), list)):
                errors.append(f"loop_melody.steps missing for {instr}.")
            else:
                if len(m["steps"]) != max(0, len(t) - 1):
                    errors.append(f"loop_melody.steps length must be len(time)-1 for {instr}.")

    # Foreground
    fg = plan.get("foreground")
    if not isinstance(fg, dict):
        errors.append("foreground missing or not an object.")
        fg = {}
    hit_instruments = fg.get("hit_instruments")
    if not (isinstance(hit_instruments, list) and len(set(hit_instruments)) == 3 and
            all(i in ORDERED_INSTRS for i in hit_instruments)):
        errors.append("foreground.hit_instruments must list exactly 3 valid instruments (explicit).")

    # disjointness vs loop
    if isinstance(hit_instruments, list) and isinstance(loop_instruments, list):
        if set(hit_instruments) & set(loop_instruments):
            errors.append("hit_instruments must be disjoint from loop_instruments.")
        if set(hit_instruments) | set(loop_instruments) != set(ORDERED_INSTRS):
            errors.append("loop_instruments ∪ hit_instruments must cover all six instruments exactly once.")

    bars = fg.get("bars")
    if not (isinstance(bars, list) and len(bars) == 3):
        errors.append("foreground.bars must be a list of exactly 3 bar objects.")
        bars = []

    for i, bar in enumerate(bars):
        evs = (bar or {}).get("events")
        if not (isinstance(evs, list) and 3 <= len(evs) <= 8):
            errors.append(f"bar {16+i}: events must be 3..8."); continue
        for j, ev in enumerate(evs):
            if not isinstance(ev, dict):
                errors.append(f"bar {16+i} event {j}: not an object."); continue
            t   = ev.get("time"); dur = ev.get("duration")
            act = ev.get("active"); pcs = ev.get("pcset")
            voh = ev.get("voicing_hint"); au = ev.get("avoid_unisons")
            if not (isinstance(t, int) and 0 <= t <= 15): errors.append(f"bar {16+i} event {j}: time must be 0..15.")
            if not (isinstance(dur, int) and 1 <= dur <= 3): errors.append(f"bar {16+i} event {j}: duration must be 1..3.")
            if not (isinstance(act, list) and 1 <= len(act) <= 3 and all(a in (hit_instruments or []) for a in act)):
                errors.append(f"bar {16+i} event {j}: active must be non-empty subset of hit_instruments.")
            if not (isinstance(pcs, list) and 2 <= len(pcs) <= 6 and all(isinstance(x, int) and 0 <= x <= 11 for x in pcs)):
                errors.append(f"bar {16+i} event {j}: pcset must be ints 0..11 (len 2..6).")
            if voh not in ("cluster","spread"): errors.append(f"bar {16+i} event {j}: voicing_hint must be 'cluster' or 'spread'.")
            if au  not in ("mild","strong"):    errors.append(f"bar {16+i} event {j}: avoid_unisons must be 'mild' or 'strong'.")

    if errors:
        raise RuntimeError("Bars 16–18 plan invalid:\n" + "\n".join(f"- {e}" for e in errors))

    return agg_validate_prepare_or_repair(agg_raw)

def request_bars1618_strict(client: OpenAI, b1_agg: dict, bar3: dict, pc456: dict, cloud789: dict, plan1014: dict,
                            attempts: int = 6, temp: float = 0.7) -> Tuple[dict, dict]:
    last_err = None

    SCHEMA_TEMPLATE = """
    {
      "aggregate_scale": {
        "name":"string", "tonic_midi":60,
        "bands":[
          {"midi_lo":28,"midi_hi":47,"pcs":[0,0,0(,0)]},
          {"midi_lo":48,"midi_hi":71,"pcs":[0,0,0(,0)]},
          {"midi_lo":72,"midi_hi":103,"pcs":[0,0,0(,0)]}
        ]
      },
      "background": {
        "loop_instruments": ["<exactly-3>"],
        "loop_rhythm": {
          "<instrA>": {"time":[...0..15...], "duration":[...]},
          "<instrB>": {"time":[...], "duration":[...]},
          "<instrC>": {"time":[...], "duration":[...]}
        },
        "loop_melody": {
          "<instrA>": {"start_hint":"low|mid|high", "steps":[...]},
          "<instrB>": {"start_hint":"low|mid|high", "steps":[...]},
          "<instrC>": {"start_hint":"low|mid|high", "steps":[...]}
        }
      },
      "foreground": {
        "hit_instruments": ["<exactly-3>"],
        "bars": [
          {"events":[
            {"time":0,"duration":1,"active":["<subset-of-hit>"],"pcset":[0,0],"voicing_hint":"cluster","avoid_unisons":"strong"}
          ]},
          {"events":[...]},
          {"events":[...]}
        ]
      },
      "meta": {"bands_count":3,"band_lengths":[3,4,5],"union_size":12}
    }
    """.strip()

    for _ in range(max(1, attempts)):
        sysF, usrF = prompt_bars1618_loop_stabs_contract(b1_agg, bar3, pc456, cloud789, plan1014)
        usrF += (
            "\nUse this exact JSON shape (no extra keys, no comments). "
            "Fill all placeholders with concrete values:\n" + SCHEMA_TEMPLATE
        )
        plan = call_llm_json(client, sysF, usrF, model="gpt-4.1", temperature=temp, top_p=0.9)
        try:
            agg = validate_bars1618_plan_or_raise(plan)
            return plan, agg
        except RuntimeError as e:
            last_err = "\n".join(line for line in str(e).splitlines() if line.startswith("- "))
            continue
    raise RuntimeError("Contract F failed validation after retries:\n" + (last_err or "(no details)"))

def realize_bars1618(plan: dict, validated_agg: dict = None) -> Dict[str, Dict[str, List[int]]]:
    """
    Realize Bars 16–18 from a validated plan (4/4, 16 ticks).
    Accepts an optional validated_agg so we don't re-parse the raw plan.
    """
    out = {instr: {"time": [], "duration": [], "pitch": [], "velocity": []} for instr in ORDERED_INSTRS}

    # Use the validated aggregate if provided; otherwise repair/validate the plan's aggregate.
    agg = validated_agg if isinstance(validated_agg, dict) else agg_validate_prepare_or_repair(plan.get("aggregate_scale", {}))

    # Background loop (one bar), repeat across 3 bars
    bg = plan["background"]
    loop_instruments = list(bg["loop_instruments"])
    loop_rhythm = bg["loop_rhythm"]; loop_melody = bg["loop_melody"]

    loop_cache = {}
    for instr in loop_instruments:
        L = 16
        r = loop_rhythm[instr]
        t_local = enforce_monophony_times([int(x) for x in r["time"] if 0 <= int(x) < L])
        d_local = [int(x) for x in r["duration"]]
        t_local, d_local = cap_durations_local(t_local, d_local, L)

        m = loop_melody[instr]
        steps = m["steps"]; start_hint = m.get("start_hint","mid")
        rng = INSTRUMENTS[instr]["range"]; tess = INSTRUMENTS[instr]["tess"]
        pitches = (pitches_from_steps_aggregate(t_local, steps, agg, rng, tess, start_hint)
                   if len(steps) == max(0, len(t_local)-1)
                   else scale_walk_assign_pitches_aggregate(t_local, agg, rng, tess))
        loop_cache[instr] = (t_local, d_local, pitches)

    bar_offsets = [BAR16_OFF, BAR17_OFF, BAR18_OFF]
    for off in bar_offsets:
        for instr in loop_instruments:
            t_local, d_local, pitches = loop_cache[instr]
            out[instr]["time"]    += absolute_times(t_local, off)
            out[instr]["duration"]+= d_local
            out[instr]["pitch"]   += pitches
            out[instr]["velocity"]+= [VEL_LOOP_BG]*len(t_local)

    # Foreground
    fg = plan["foreground"]
    hit_instruments = list(fg["hit_instruments"])
    bars = fg["bars"]

    prev_pitches = {}
    for idx, bar in enumerate(bars):
        off = bar_offsets[idx]
        events = sorted(bar["events"], key=lambda e: int(e["time"]))
        for ev in events:
            t = int(ev["time"]); dur = int(ev["duration"])
            active = [i for i in ev["active"] if i in hit_instruments]
            pcset = [int(x) % 12 for x in ev["pcset"]]
            voh = ev.get("voicing_hint","cluster")
            au  = ev.get("avoid_unisons","strong")
            regs = ev.get("register_targets", {}) or {}
            picks = realize_pcset_event_subset(pcset, voh, regs, au, active, prev_pitches, agg=agg)
            for instr, pitch in picks.items():
                out[instr]["time"].append(off + t)
                out[instr]["duration"].append(dur)
                out[instr]["pitch"].append(int(pitch))
                out[instr]["velocity"].append(VEL_FORE_STAB)
                prev_pitches[instr] = int(pitch)

    return out

def collect_pitches_in_window(per_instr_events: Dict[str, Dict[str, List[int]]],
                              t_lo: int, t_hi: int) -> Dict[str, List[int]]:
    """Unique MIDI pitches per instrument whose ONSETS fall in [t_lo, t_hi)."""
    out = {}
    for instr in ORDERED_INSTRS:
        t = per_instr_events[instr]["time"]
        p = per_instr_events[instr]["pitch"]
        picks = sorted({int(p[i]) for i in range(len(t)) if t_lo <= int(t[i]) < t_hi})
        out[instr] = picks
    return out

def prompt_bar19_recap_chord(candidates: Dict[str, List[int]]) -> Tuple[str, str]:
    system = """
You output ONLY strict JSON. No prose.

Return:
{ "bar19_chord": {
  "alto_flute": int, "violin": int, "bass_clarinet": int,
  "trumpet": int, "cello": int, "double_bass": int
}}

Hard constraints:
- For EACH instrument, you MUST choose exactly ONE MIDI pitch from the provided candidate list for that instrument.
- Choose a sonority with rich 3rds/4ths across voices (tertiary/quartal flavor), but DO NOT invent pitches.
- Avoid exact pitch duplicates if possible; octave doublings are acceptable.
Validate internally and only return a valid JSON object.
"""
    user = "Candidates per instrument (MIDI, allowed set to choose from):\n" + json.dumps(candidates, indent=2)
    return system.strip(), user.strip()

def choose_bar19_chord_via_llm(client: OpenAI,
                               candidates: Dict[str, List[int]]) -> Dict[str, int]:
    sysR, usrR = prompt_bar19_recap_chord(candidates)
    js = call_llm_json(client=client, system_msg=sysR, user_msg=usrR,
                       model="gpt-4.1", temperature=0.4, top_p=0.9)
    chord = (js.get("bar19_chord") or {})
    # Validate strictly: if LLM slips, pick a safe fallback (mid candidate or tess-center snap)
    out = {}
    for instr in ORDERED_INSTRS:
        opts = candidates.get(instr, [])
        pick = chord.get(instr)
        if isinstance(pick, int) and pick in opts:
            out[instr] = int(pick)
        else:
            if opts:
                out[instr] = int(opts[len(opts)//2])
            else:
                # last resort: snap to tess center
                lo, hi = INSTRUMENTS[instr]["tess"]
                out[instr] = clamp((lo+hi)//2, INSTRUMENTS[instr]["range"][0], INSTRUMENTS[instr]["range"][1])
    return out

def request_jazzy_aggregates(client: OpenAI, prior_aggs: List[dict], k: int = 3) -> List[dict]:
    """Request k fresh jazzy aggregates, each diverse vs all priors and earlier picks."""
    picks = []
    priors = list(prior_aggs)
    for idx in range(max(1, k)):
        def mk():
            return prompt_jazzy_aggregate(f"group{idx+1}", [pc_to_band_map(a) for a in priors if a.get("bands")])
        js, agg = request_fresh_aggregate(
            make_prompts_fn=mk, key_in_json="aggregate_scale",
            client=client, prior_aggs=priors, attempts=5, temp=1.15
        )
        picks.append(agg); priors.append(agg)
    return picks

def realize_longnote_texture_20_25(jazz_aggs: List[dict], seed: int = None) -> Dict[str, Dict[str, List[int]]]:
    """
    Bars 20–25 (each 12 ticks, 3/4): slowly changing long-note texture.
    - Few, long sustains per instrument, randomly introduced, overlapping freely (no barline clipping).
    - Harmony comes from 3 jazzy aggregates that change every two bars: [20–21]=A, [22–23]=B, [24–25]=C.
    """
    rng = random.Random(seed)
    aggA = jazz_aggs[0]; aggB = jazz_aggs[1 if len(jazz_aggs) > 1 else 0]; aggC = jazz_aggs[2 if len(jazz_aggs) > 2 else 1 if len(jazz_aggs) > 1 else 0]
    region_lo, region_hi = BAR20_OFF, BAR25_OFF + 12

    def agg_for_time(tt: int):
        if tt < BAR22_OFF: return aggA
        if tt < BAR24_OFF: return aggB
        return aggC

    out = {instr: {"time": [], "duration": [], "pitch": [], "velocity": []} for instr in ORDERED_INSTRS}

    # density per instrument (slow texture)
    base_counts = {
        "double_bass":   3,  # across 6 bars
        "cello":         3,
        "bass_clarinet": 4,
        "trumpet":       3,
        "violin":        4,
        "alto_flute":    4,
    }

    for instr in ORDERED_INSTRS:
        lo, hi = INSTRUMENTS[instr]["range"]; tess = INSTRUMENTS[instr]["tess"]
        n = max(1, base_counts[instr] + rng.choice([-1,0,0,0,1]))
        # random start times across region
        starts = []
        for _ in range(n * 2):  # oversample, then trim to n unique
            # choose bar 20..25 uniformly, onset within bar mostly in first 2 beats
            bar_start = rng.choice([BAR20_OFF, BAR21_OFF, BAR22_OFF, BAR23_OFF, BAR24_OFF, BAR25_OFF])
            inbar = rng.choice([0,1,2,3,4,5,6,7,8,9])  # keep some headroom
            starts.append(bar_start + inbar)
        starts = sorted(sorted(set(starts))[:n])

        prev_pc = None
        for st in starts:
            if st >= region_hi: continue
            # long-ish durations, biased longer; cap to region end later by safety pass
            dur = rng.choice([6,8,9,10,12,14,16,18])
            # choose aggregate for this onset
            agg = agg_for_time(st)
            center = (tess[0] + tess[1])//2 + rng.choice([-2,-1,0,0,1,2])
            bi = agg_find_band_for_midi(center, agg["bands"])
            pcs = list(agg["bands"][bi]["pcs"])
            pc = pcs[rng.randrange(len(pcs))] if prev_pc is None else _pick_pc_close_to(prev_pc, pcs)
            pitch = nearest_pitch_for_pc_aggregate(pc, center, (lo,hi), agg)
            vel = clamp(VEL_LONGTEXT_BASE + rng.choice([-8,-6,-4,-2,0,0,0,2,4,6,8]), 40, 110)

            out[instr]["time"].append(st)
            out[instr]["duration"].append(dur)
            out[instr]["pitch"].append(int(pitch))
            out[instr]["velocity"].append(vel)
            prev_pc = pc

    return out

def realize_longnote_texture_26_30(base_agg: dict, seed: int = None) -> Dict[str, Dict[str, List[int]]]:
    rng = random.Random(seed)
    out = {instr: {"time": [], "duration": [], "pitch": [], "velocity": []} for instr in ORDERED_INSTRS}

    window_start = BAR26_OFF
    window_end   = BAR30_OFF + BAR30_TICKS  # 368

    # Chair-based center offsets (kept consistent with earlier voicing choices)
    centers = {
        "double_bass": -12, "cello": -7, "bass_clarinet": -3,
        "trumpet": +2, "violin": +7, "alto_flute": +9,
    }

    for instr in ORDERED_INSTRS:
        lo, hi = INSTRUMENTS[instr]["range"]
        tess_lo, tess_hi = INSTRUMENTS[instr]["tess"]
        center_guess = (tess_lo + tess_hi)//2 + centers.get(instr, 0)

        # 2–3 long sustains per instrument across bars 26–30
        n = 2 + rng.randrange(2)  # 2 or 3 segments
        starts = [window_start + rng.randint(0, max(1, (window_end - window_start) - 10)) for _ in range(n)]

        # seam anchors to guarantee continuity right at bar 26
        anchor_leaders = {"double_bass", "violin"}
        if instr in anchor_leaders and not any(window_start <= s <= window_start + 2 for s in starts):
            starts.append(window_start + rng.choice([0, 1, 2]))

        starts = sorted(set(starts))
        last_pc = None

        for s in starts:
            # 8–18 ticks sustain, clipped to the window
            dur = rng.randint(8, 18)
            end = min(window_end, s + dur)
            dur = max(1, end - s)

            # Pick a band by the center, then a 'jazzy-near' pc relative to previous
            bi = agg_find_band_for_midi(center_guess, base_agg["bands"])
            band_pcs = list(base_agg["bands"][bi]["pcs"])
            if last_pc is None:
                target_pc = band_pcs[rng.randrange(len(band_pcs))]
            else:
                target_pc = _pick_pc_close_to(last_pc, band_pcs)

            pitch = nearest_pitch_for_pc_aggregate(int(target_pc), center_guess + rng.choice([-2, -1, 0, 1, 2]),
                                                   (lo, hi), base_agg)

            out[instr]["time"].append(int(s))
            out[instr]["duration"].append(int(dur))
            out[instr]["pitch"].append(int(pitch))
            out[instr]["velocity"].append(VEL_JAZZ_CH2)

            last_pc = int(target_pc)

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

def realize_pcset_event_subset(pcset: List[int], voicing_hint: str, reg_targets: Dict[str,str],
                               avoid_unisons: str, active: List[str],
                               prev_pitches: Dict[str,int] = None,
                               agg: dict = None) -> Dict[str,int]:
    """
    Like realize_pcset_event, but only voices for a subset of instruments `active`.
    Returns {instr: pitch} for instruments in `active` only.
    """
    out = {}
    taken = set()
    ordered_active = [i for i in ORDERED_INSTRS if i in active]

    for instr in ordered_active:
        lo, hi = INSTRUMENTS[instr]["range"]
        tess = INSTRUMENTS[instr]["tess"]
        hint = (reg_targets or {}).get(instr, "mid")
        center = _center_from_register_hint(tess, hint)
        approx = _spread_bias(instr, center, voicing_hint)

        # keep common tone if possible
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

        # pick pitch honoring aggregate bands when provided
        if agg is not None:
            pick = nearest_pitch_for_pc_aggregate(int(target_pc), approx, (lo,hi), agg)
        else:
            pick = _nearest_pitch_for_pcset(int(target_pc), approx, (lo,hi))

        # avoid exact duplicates if "strong" and possible
        if avoid_unisons == "strong" and pick in out.values() and len(pcset) > 1:
            for alt_pc in pcset:
                if alt_pc == target_pc: continue
                q = (nearest_pitch_for_pc_aggregate(int(alt_pc), approx, (lo,hi), agg)
                     if agg is not None else _nearest_pitch_for_pcset(int(alt_pc), approx, (lo,hi)))
                if q not in out.values():
                    pick = q
                    target_pc = alt_pc
                    break

        out[instr] = int(pick)
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
    # ---- NEW: Bar 2 = two *free* aggregates (no degree logic) ----
    seedH = random.randint(1, 10**9)
    agg2A, agg2B = request_bar2_two_free_aggregates(oai, b1_agg)
    bar2_parts = bar2_two_hits_from_free_aggs(agg2A, agg2B, seed=seedH)

    # Optional: feed these into prior-aggs lists later for extra diversity pressure
    extra_bar2_aggs = [agg2A, agg2B]

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
    sysC, usrC = prompt_bars456_pcsets_contract(b1_agg, bar3)
    seedC = random.randint(1, 10**9)
    usrC += f"\nCreativeSeed: {seedC}\nRule: When several valid pcsets/hints fit, bias choices using CreativeSeed for diversity."
    
    def mk_prompts_C():
        return sysC, usrC
    pc456, agg456 = request_fresh_aggregate(
        make_prompts_fn=mk_prompts_C,
        key_in_json="aggregate_scale",
        client=oai,
        prior_aggs=[b1_agg, contrast_agg] + extra_bar2_aggs,
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
    sysD, usrD = prompt_bars789_clouds_contract(b1_agg, bar3, pc456)
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

    # ---- LLM Contract E (Bars 10–14: background loop + foreground stabs) ----
    sysE, usrE = prompt_bars1014_loop_stabs_contract(b1_agg, bar3, pc456, cloud789)
    seedE = random.randint(1, 10**9)
    usrE += f"\nCreativeSeed: {seedE}\nRule: Use CreativeSeed to break ties (loop selection, rhythms, events)."

    def mk_prompts_E():
        return sysE, usrE

    plan1014, agg1014 = request_bars1014_strict(
        client=oai,
        b1_agg=b1_agg,
        bar3=bar3,
        pc456=pc456,
        cloud789=cloud789,
        attempts=8,
        temp=0.6
    )
    bars1014 = realize_bars1014(plan1014, agg1014)

    # Append Bars 10–14 to per-instrument accumulators
    for instr in ORDERED_INSTRS:
        part = bars1014[instr]
        per_instr_events[instr]["time"]    += part["time"]
        per_instr_events[instr]["duration"]+= part["duration"]
        per_instr_events[instr]["pitch"]   += part["pitch"]
        per_instr_events[instr]["velocity"]+= part["velocity"]

    # ---- Bar 15: exact repetition of Bar 2 (2/4), appended at end ----
    SHIFT_2_TO_15 = BAR15_OFF - BAR2_OFF  # 176 - 12 = 164
    for instr in ORDERED_INSTRS:
        part2 = bar2_parts[instr]  # already built earlier
        per_instr_events[instr]["time"]    += [t + SHIFT_2_TO_15 for t in part2["time"]]
        per_instr_events[instr]["duration"]+= part2["duration"][:]
        per_instr_events[instr]["pitch"]   += part2["pitch"][:]
        per_instr_events[instr]["velocity"]+= part2["velocity"][:]

    # ---- LLM Contract F (Bars 16–18: background loop + foreground stabs, 4/4) ----
    sysF, usrF = prompt_bars1618_loop_stabs_contract(b1_agg, bar3, pc456, cloud789, plan1014)
    seedF = random.randint(1, 10**9)
    usrF += f"\nCreativeSeed: {seedF}\nRule: Use CreativeSeed to break ties (loop selection, rhythms, events)."

    plan1618, agg1618 = request_bars1618_strict(
        client=oai,
        b1_agg=b1_agg,
        bar3=bar3,
        pc456=pc456,
        cloud789=cloud789,
        plan1014=plan1014,
        attempts=8,
        temp=0.6
    )
    bars1618 = realize_bars1618(plan1618, agg1618)

    # Append Bars 16–18 to per-instrument accumulators
    for instr in ORDERED_INSTRS:
        part = bars1618[instr]
        per_instr_events[instr]["time"]    += part["time"]
        per_instr_events[instr]["duration"]+= part["duration"]
        per_instr_events[instr]["pitch"]   += part["pitch"]
        per_instr_events[instr]["velocity"]+= part["velocity"]

    # ---- Bar 19: 3/4 recap chord on beat 1 (quarter-note), chosen by LLM from Bars 16–18 ----
    # Collect candidates from onsets in bars 16–18
    cand_16_18 = collect_pitches_in_window(per_instr_events, BAR16_OFF, BAR19_OFF)
    chord19 = choose_bar19_chord_via_llm(oai, cand_16_18)

    for instr in ORDERED_INSTRS:
        per_instr_events[instr]["time"].append(BAR19_OFF + 0)       # beat 1
        per_instr_events[instr]["duration"].append(4)               # quarter note
        per_instr_events[instr]["pitch"].append(int(chord19[instr]))
        per_instr_events[instr]["velocity"].append(VEL_BAR19_RECAP)

        # ---- Bars 20–30: one-pass long-note texture (no seams, single generator) ----
        prior_aggs_for_jazz = [b1_agg, contrast_agg, agg456, agg789, agg1014, agg1618]
        jazzy_aggs = request_jazzy_aggregates(oai, prior_aggs_for_jazz, k=3)

        seedJ = random.randint(1, 10**9)
        longtex_20_30 = realize_longnote_texture_20_30(jazzy_aggs, seed=seedJ)

        for instr in ORDERED_INSTRS:
            part = longtex_20_30[instr]
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

        # enforce absolute monophony by capping ONLY to next onset (no barline clipping)
        t2, d2, p2, v2 = [], [], [], []
        for i in range(len(t)):
            tt = t[i]; dd = int(d[i]); pp = int(p[i]); vv = int(v[i])
            if i < len(t)-1:
                dd = min(dd, max(0, t[i+1] - tt))
            dd = min(dd, GLOBAL_END - tt)
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
            {"meter":[3,4],"start":36},
            {"meter":[4,4],"start":48},
            {"meter":[4,4],"start":64},
            {"meter":[3,4],"start":80},
            {"meter":[3,4],"start":92},
            {"meter":[3,4],"start":104},
            {"meter":[3,4],"start":116},
            {"meter":[3,4],"start":128},
            {"meter":[3,4],"start":140},
            {"meter":[3,4],"start":152},
            {"meter":[3,4],"start":164},
            {"meter":[2,4],"start":176},
            {"meter":[4,4],"start":184},  # 16
            {"meter":[4,4],"start":200},  # 17
            {"meter":[4,4],"start":216},  # 18
            {"meter":[1,4],"start":232},  # 19
            {"meter":[3,4],"start":236},  # 20
            {"meter":[3,4],"start":248},  # 21
            {"meter":[3,4],"start":260},  # 22
            {"meter":[3,4],"start":272},  # 23
            {"meter":[3,4],"start":284},  # 24
            {"meter":[3,4],"start":296},  # 25
            {"meter":[3,4],"start":308},  # 26
            {"meter":[3,4],"start":320},  # 27
            {"meter":[3,4],"start":332},  # 28
            {"meter":[3,4],"start":344},  # 29
            {"meter":[3,4],"start":356}   # 30
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
