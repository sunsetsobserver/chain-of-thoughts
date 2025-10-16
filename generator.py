#!/usr/bin/env python3

import os, sys, time, json, random, requests
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

# --- helper: compress first fragment plan so LLM won’t copy it verbatim ---
def _compress_plan(bars_obj: dict) -> dict:
    out = []
    for i in range(4):
        b = bars_obj["bars"][i]
        out.append({
            "rhythm": {k: {"time": v.get("time", []), "duration": v.get("duration", [])}
                       for k, v in (b.get("rhythm", {}) or {}).items()},
            "melody": {k: {"steps": (b.get("melody", {}) or {}).get(k, {}).get("steps", [])}
                       for k in ORDERED_INSTRS if k in (b.get("melody", {}) or {})}
        })
    return {"bars": out}


# ---------------- Musical constants ----------------

BAR1_TICKS = 12    # 3/4
BAR1_OFF   = 0
BAR2_TICKS = 8     # 2/4
BAR2_OFF   = 12
BAR3_TICKS = 16    # 4/4
BAR3_OFF   = 20
BAR4_TICKS = 12    # 3/4
BAR4_OFF   = 36
BAR5_TICKS = 16    # 4/4
BAR5_OFF   = 48
BAR6_TICKS = 16    # 4/4
BAR6_OFF   = 64
BAR7_TICKS = 12    # 3/4
BAR7_OFF   = 80
BAR8_TICKS = 12    # 3/4
BAR8_OFF   = 92
BAR9_TICKS = 12    # 3/4
BAR9_OFF   = 104
BAR10_TICKS = 12   # 3/4
BAR10_OFF   = 116
BAR11_TICKS = 12   # 3/4
BAR11_OFF   = 128
BAR12_TICKS = 12   # 3/4
BAR12_OFF   = 140
BAR13_TICKS = 12   # 3/4
BAR13_OFF   = 152
BAR14_TICKS = 12   # 3/4
BAR14_OFF   = 164
BAR15_TICKS = 8    # 2/4
BAR15_OFF   = 176
BAR16_TICKS = 16   # 4/4
BAR16_OFF   = 184
BAR17_TICKS = 16   # 4/4
BAR17_OFF   = 200
BAR18_TICKS = 16   # 4/4
BAR18_OFF   = 216
BAR19_TICKS = 4    # 1/4 (recap hit)
BAR19_OFF   = 232
BAR20_TICKS = 4    # 1/4 (short break)
BAR20_OFF   = 236
BAR21_TICKS = 16   # 4/4
BAR21_OFF   = 240
BAR22_TICKS = 16   # 4/4
BAR22_OFF   = 256
BAR23_TICKS = 16   # 4/4
BAR23_OFF   = 272
BAR24_TICKS = 16   # 4/4
BAR24_OFF   = 288
BAR25_TICKS = 16   # 4/4
BAR25_OFF   = 304
BAR26_TICKS = 16   # 4/4
BAR26_OFF   = 320
BAR27_TICKS = 16   # 4/4
BAR27_OFF   = 336
BAR28_TICKS = 16   # 4/4
BAR28_OFF   = 352
BAR29_TICKS = 16   # 4/4
BAR29_OFF   = 368
BAR30_TICKS = 16   # 4/4
BAR30_OFF   = 384
BAR31_TICKS = 16   # 4/4
BAR31_OFF   = 400
BAR32_TICKS = 16   # 4/4
BAR32_OFF   = 416
BAR33_TICKS = 16   # 4/4
BAR33_OFF   = 432
BAR34_TICKS = 16   # 4/4
BAR34_OFF   = 448
BAR35_TICKS = 12   # 3/4
BAR35_OFF   = 464
BAR36_TICKS = 16   # 4/4
BAR36_OFF   = 476
BAR37_TICKS = 16   # 4/4
BAR37_OFF   = 492
BAR38_TICKS = 12   # 3/4
BAR38_OFF   = 508
BAR39_TICKS = 16   # 4/4
BAR39_OFF   = 520
BAR40_TICKS = 16   # 4/4
BAR40_OFF   = 536
BAR41_TICKS = 4    # 1/4 short break
BAR41_OFF   = 552
BAR42_TICKS = 12   # 3/4
BAR42_OFF   = 556
BAR43_TICKS = 12   # 3/4
BAR43_OFF   = 568
BAR44_TICKS = 12   # 3/4
BAR44_OFF   = 580
BAR45_TICKS = 12   # 3/4
BAR45_OFF   = 592
BAR46_TICKS = 12
BAR46_OFF   = 604
BAR47_TICKS = 12
BAR47_OFF   = 616
BAR48_TICKS = 12
BAR48_OFF   = 628
BAR49_TICKS = 12
BAR49_OFF   = 640
BAR50_TICKS = 16
BAR50_OFF   = 652

# Move the global end to the end of bar 49
GLOBAL_END  = 668

VEL_BAR2_FIRST = 124
VEL_BAR2_SECOND = 114
VEL_BAR3_PP = 30
VEL_LOOP_BG = 48       # background loop dynamic
VEL_FORE_STAB = 92     # foreground staccato hits dynamic
VEL_BAR19_RECAP = 104     # mf–f recap hit
VEL_DRONE = 64
VEL_DRONE_HARM = 76  # chordal additions during bars 21–34 (a touch brighter than VEL_DRONE)
VEL_PPP = 22           # very, very soft for Bars 42–45 (all except trumpet)
VEL_TRUMPET_LOUD = 110 # loud trumpet line for Bars 42–44
VEL_RIPPLE = 88
VEL_TUTTI_END = 112  # loud dynamic for the two concluding tutti chords at the end of Part 1

# Ranges (sounding MIDI)
INSTRUMENTS = {
    "alto_flute":    {"range": (53, 81),  "tess": (55, 79)},
    "violin":        {"range": (55, 88), "tess": (60, 84)},
    "bass_clarinet": {"range": (43, 74),  "tess": (41, 70)},
    "trumpet":       {"range": (60, 82),  "tess": (60, 78)},
    "cello":         {"range": (48, 74),  "tess": (50, 69)},
    "double_bass":   {"range": (31, 55),  "tess": (33, 50)},
}

ORDERED_INSTRS = ["alto_flute","violin","bass_clarinet","trumpet","cello","double_bass"]

INSTRUMENT_META = {
    "alto_flute":    {"display_name": "Alto Flute",         "gm_program": 73, "bank": 0, "transpose": 0, "clef": "treble"},
    "violin":        {"display_name": "Violin",             "gm_program": 40, "bank": 0, "transpose": 0, "clef": "treble"},
    "bass_clarinet": {"display_name": "Bass Clarinet in Bb","gm_program": 71, "bank": 0, "transpose": 0, "clef": "bass"},
    "trumpet":       {"display_name": "Trumpet in C",       "gm_program": 56, "bank": 0, "transpose": 0, "clef": "treble"},
    "cello":         {"display_name": "Cello",              "gm_program": 42, "bank": 0, "transpose": 0, "clef": "bass"},
    "double_bass":   {"display_name": "Double Bass",        "gm_program": 43, "bank": 0, "transpose": 0, "clef": "bass"},
}

# Dynamics (fixed)
def vel_bar1_p_cresc(local_t: int) -> int:
    # 50 → 80 over ticks 0..11
    return max(0, min(127, round(50 + (80-50) * (local_t / 11.0))))

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

def compute_next_bar_offset(per_instr_events: Dict[str, Dict[str, List[int]]], bar_size: int = 16) -> int:
    """Return the absolute tick of the next bar boundary after all currently scheduled events."""
    max_end = 0                                               # track the furthest end position among all events
    for instr in ORDERED_INSTRS:                              # iterate through all instruments in a fixed order
        T = per_instr_events[instr]["time"]                   # absolute onset times for this instrument
        D = per_instr_events[instr]["duration"]               # durations (in ticks) for this instrument
        for i in range(len(T)):                               # scan each event for this instrument
            max_end = max(max_end, int(T[i]) + int(D[i]))     # update furthest end (onset + duration)
    # snap up to the next multiple of bar_size, so we start on a clean bar boundary
    return ((max_end + bar_size - 1) // bar_size) * bar_size  # ceiling division * bar_size

def collect_recent_pcs(per_instr_events: Dict[str, Dict[str, List[int]]], lookback_bars: int = 8, bar_size: int = 16) -> List[int]:
    """Return a sorted list of unique pitch-classes (0..11) sounding in the last `lookback_bars` bars."""
    end_tick = compute_next_bar_offset(per_instr_events, bar_size=bar_size)  # end of the current music (rounded to bar)
    start_tick = max(0, end_tick - lookback_bars * bar_size)                 # look back a fixed number of bars
    pcs = set()                                                              # accumulate unique pitch-classes here
    for instr in ORDERED_INSTRS:                                             # scan all instruments
        T = per_instr_events[instr]["time"]                                  # their onsets (absolute ticks)
        D = per_instr_events[instr]["duration"]                               # their durations (ticks)
        P = per_instr_events[instr]["pitch"]                                  # their pitches (MIDI)
        for i in range(len(T)):                                              # visit each event
            t0 = int(T[i])                                                   # event start
            t1 = t0 + int(D[i])                                              # event end
            if not (t1 <= start_tick or t0 >= end_tick):                     # keep if event overlaps the window
                pcs.add(int(P[i]) % 12)                                      # store pitch-class modulo 12
    return sorted(pcs)                                                       # return sorted unique pcs for readability


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
    elif t < 236:   return 1, 4   # bar 19 (recap hit)
    elif t < 240:   return 1, 4   # bar 20 (short break)
    elif t < 256:   return 4, 4   # bar 21
    elif t < 272:   return 4, 4   # bar 22
    elif t < 288:   return 4, 4   # bar 23
    elif t < 304:   return 4, 4   # bar 24
    elif t < 320:   return 4, 4   # bar 25
    elif t < 336:   return 4, 4   # bar 26
    elif t < 352:   return 4, 4   # bar 27
    elif t < 368:   return 4, 4   # bar 28
    elif t < 384:   return 4, 4   # bar 29
    elif t < 400:   return 4, 4   # bar 30
    elif t < 416:   return 4, 4   # bar 31
    elif t < 432:   return 4, 4   # bar 32
    elif t < 448:   return 4, 4   # bar 33
    elif t < 464:   return 4, 4   # bar 34
    elif t < 476:   return 3, 4   # bar 35 (repeat of bar 4)
    elif t < 492:   return 4, 4   # bar 36 (repeat of bar 5)
    elif t < 508:   return 4, 4   # bar 37 (repeat of bar 6)
    elif t < 520:   return 3, 4   # bar 38 (variation of bar 4)
    elif t < 536:   return 4, 4   # bar 39 (variation of bar 5)
    elif t < 552:   return 4, 4   # bar 40 (variation of bar 6)
    elif t < 556:   return 1, 4   # bar 41 short break
    elif t < 568:   return 3, 4   # bar 42
    elif t < 580:   return 3, 4   # bar 43
    elif t < 592:   return 3, 4   # bar 44
    elif t < 604:   return 3, 4   # bar 45
    elif t < 616: return 3, 4   # bar 46
    elif t < 628: return 3, 4   # bar 47
    elif t < 640: return 3, 4   # bar 48
    elif t < 652: return 3, 4   # bar 49 / global end
    elif t < 668:  return 4, 4
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

def choose_trumpet_long_notes_via_llm(oai, agg, seed, instr="trumpet", avoid_midis=None):
    """
    Return 3 absolute MIDI notes for the trumpet:
    - within instrument range
    - pitch classes from the aggregate's pcs
    - different from avoid_midis if provided
    - include at least one leap >= 3 semitones
    Falls back to a safe pattern if the LLM slips.
    """
    import json, random
    rng  = INSTRUMENTS[instr]["range"]
    tess = INSTRUMENTS[instr]["tess"]
    pcs_union = sorted({pc for b in agg.get("bands", []) for pc in b.get("pcs", [])}) or [0,4,7]

    system = "You output ONLY strict JSON with a single key 'pitches' (array of 3 integers). No prose."
    user = {
        "task": "choose-three-long-notes",
        "instrument": instr,
        "range_midi": list(rng),
        "tessitura_hint": list(tess),
        "allowed_pitch_classes": pcs_union,
        "avoid_midis": [int(x) for x in (avoid_midis or [])],
        "hard_rules": [
            "Return exactly three integers in 'pitches'.",
            "All pitches must be within the given range.",
            "Each pitch-class (mod 12) must be in allowed_pitch_classes.",
            "Include at least one leap of 3 semitones or more.",
            "If avoid_midis is non-empty, do not return the exact same triple."
        ],
        "creative_seed": int(seed)
    }
    try:
        resp = call_llm_json(oai, system, json.dumps(user), model="gpt-4.1", temperature=1.05)
        cand = resp.get("pitches", [])
        out = []
        for p in cand[:3]:
            try:
                p = int(p)
            except:
                continue
            if rng[0] <= p <= rng[1] and (p % 12) in pcs_union:
                out.append(p)
        if len(out) == 3:
            return out
    except Exception:
        pass

    # Fallback: center-ish pattern with a leap
    center = (tess[0] + tess[1]) // 2
    rnd = random.Random(seed)
    pcs = pcs_union[:]; rnd.shuffle(pcs); pcs = (pcs + pcs)[:3]
    approx_seq = [center - 2, center + 3, center - 5]  # includes a leap
    out = []
    for approx, pc in zip(approx_seq, pcs):
        out.append(nearest_pitch_for_pc_aggregate(pc, clamp(approx, rng[0], rng[1]), rng, agg))
    return [int(x) for x in out]

def request_part1_finalbar_twochords_jazz(client, recent_pcs: List[int], temp: float = 0.9) -> dict:
    """
    Ask the LLM for two complex, suspended-sounding jazz chords (as pitch-classes) to end Part 1.
    The LLM is told to avoid simple dominant→tonic cadences and to keep things harmonically rich.
    """
    # Build a concise system prompt describing the task role
    system_msg = (
        "You are a jazz harmony assistant for concert music. "
        "Given recent pitch-classes (0=C,1=C#/Db,...,11=B), propose TWO final chords for a tutti hit bar. "
        "Avoid plain dominant→tonic cadences; prefer suspended colors (e.g., lydian dominant, altered, upper-structure stacks). "
        "Return only valid JSON."
    )
    # Compose a user message that provides context + strict output schema
    user_msg = {
        "instruction": "Design two final jazz chords for a single 4/4 bar with two hits (beat 1 and beat 3).",
        "recent_pitch_classes": recent_pcs,  # give the harmonic context to keep continuity
        "requirements": {
            "no_plain_cadence": True,       # explicitly avoid V→I or any obvious resolution
            "chord1_and_chord2": "arrays of unique integers in [0..11], length between 5 and 8",
            "flavor": "suspended/altered/lydian dominant/upper-structure; leave the form open-ended",
            "voice_leading": "let chord2 be related but not resolving; small shared subset acceptable"
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "chord1": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 11}},
                "chord2": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 11}},
                "labels": {"type": "object"}  # optional names/comments; ignored by validator if absent
            },
            "required": ["chord1", "chord2"]
        }
    }
    # Call your existing JSON LLM helper (same as elsewhere in your file)
    js = call_llm_json(
        client=client,                     # OpenAI client that you already use
        system_msg=system_msg,             # system role
        user_msg=json.dumps(user_msg),     # pass user content as a JSON string for clarity
        model="gpt-4.1",                   # keep consistent with your other sections
        temperature=temp,                  # a bit adventurous to get colorful voicings
        top_p=0.9                          # mild nucleus sampling as in your other calls
    )
    return js                              # hand raw JSON back to a validator


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
            rng = INSTRUMENTS[instr]["tess"];  tess = INSTRUMENTS[instr]["tess"]
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

def validate_twochords_jazz_payload(js: dict) -> Tuple[List[int], List[int]]:
    """
    Validate and sanitize the LLM output for the two-chord ending.
    Ensures: arrays exist, ints 0..11, unique, length in 5..8. Fallback uses colorful, non-cadential pcs.
    """
    def norm(arr) -> List[int]:
        """Normalize a candidate array: ints 0..11, unique, clipped size."""
        if not isinstance(arr, list):                       # require list
            return []                                       # else reject
        out = []                                            # build a sanitized list
        seen = set()                                        # track uniqueness
        for x in arr:                                       # scan each element
            try:
                v = int(x) % 12                             # coerce to int & modulo 12
            except Exception:
                continue                                    # skip anything non-integer
            if v not in seen:                               # enforce uniqueness
                seen.add(v)                                 # mark as seen
                out.append(v)                               # append to output
        # clamp length between 5 and 8 to ensure rich voicings without bloat
        if len(out) < 5:                                    # too few → reject
            return []
        if len(out) > 8:                                    # too many → trim
            out = out[:8]
        return out

    c1 = norm(js.get("chord1", []))                         # sanitize first chord pcs
    c2 = norm(js.get("chord2", []))                         # sanitize second chord pcs
    if c1 and c2:                                           # if both look valid
        return c1, c2                                       # accept them
    # Fallback: handcraft two suspended/altered pc-sets with overlap but no clear cadence
    # (Example flavors: lydian dominant on tritone apart + altered with #11/13 color)
    fallback_c1 = [0, 2, 4, 6, 9, 10]                       # {C,D,E,F#,A,A#} → C13(#11no3) flavor
    fallback_c2 = [1, 3, 6, 8, 10]                          # {C#,D#,F#,G#,A#} → Db7(#11no5) / tritone-ish
    return fallback_c1, fallback_c2                          # guaranteed suspended, non-cadential

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
        rng = INSTRUMENTS[instr]["tess"];  tess = INSTRUMENTS[instr]["tess"]
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

def realize_part1_finalbar_tutti(
    per_instr_events: Dict[str, Dict[str, List[int]]],   # global accumulator (mutated)
    chord1_pcs: List[int],                               # first chord as pitch-classes 0..11
    chord2_pcs: List[int],                               # second chord as pitch-classes 0..11
    bar_off: int,                                        # absolute start tick for this final bar
    bar_size: int = 16,                                  # assume 4/4 mapped to 16 ticks
    hit_positions: Tuple[int, int] = (0, 8),             # two hits at beat 1 and beat 3 (0 and 8)
    hit_duration: int = 6                                # ring each hit for 6 ticks (leaves 2-tick air)
) -> None:
    """Write two tutti chord hits across all instruments at the given bar offset."""
    # Local helper: nearest in-range pitch for a given pitch-class (try to stay near tessitura center)
    def nearest_pitch_for_pc(pc: int, instr: str, approx: int) -> int:
        lo, hi = INSTRUMENTS[instr]["range"]             # absolute playable range for instrument
        best, bestd = approx, 10**9                      # track nearest MIDI note and distance
        for k in range(-24, 25):                         # search ± 2 octaves around the approx
            cand = clamp(approx + k, lo, hi)             # clamp into instrument range
            if cand % 12 == pc:                          # check pitch-class match
                d = abs(cand - approx)                   # compute distance from approx
                if d < bestd:                            # keep nearest solution
                    best, bestd = cand, d                # update best candidate
        return best                                      # return the nearest pitch with the given pc

    # Precompute tessitura centers for stable voicing placement per instrument
    tess_center = {instr: (INSTRUMENTS[instr]["tess"][0] + INSTRUMENTS[instr]["tess"][1]) // 2
                   for instr in ORDERED_INSTRS}          # mid of tessitura to guide voicing heights

    # First hit: assign each instrument a chord1 tone near its tessitura center
    chord1_pcs_cycle = list(chord1_pcs)                  # copy list to rotate through pcs
    # choose a deterministic rotation so neighboring instruments don’t always get same pcs
    for idx, instr in enumerate(ORDERED_INSTRS):         # visit instruments in the fixed orchestral order
        if not chord1_pcs_cycle:                         # safety: if empty (shouldn’t happen), skip writing
            continue                                     # nothing to voice
        pc = chord1_pcs_cycle[idx % len(chord1_pcs_cycle)]  # round-robin select a pc from chord1
        approx = tess_center[instr]                      # aim around the tessitura center
        pitch = nearest_pitch_for_pc(pc, instr, approx)  # snap to the nearest in-range note with that pc
        per_instr_events[instr]["time"].append(bar_off + hit_positions[0])    # onset at the first hit
        per_instr_events[instr]["duration"].append(hit_duration)              # sustain for chosen duration
        per_instr_events[instr]["pitch"].append(int(pitch))                   # write realized MIDI pitch
        per_instr_events[instr]["velocity"].append(VEL_TUTTI_END)             # loud tutti dynamic

    # Second hit: related but not resolving — use chord2 pcs and try small voice-leading motion
    chord2_pcs_cycle = list(chord2_pcs)                  # copy to rotate independently
    for idx, instr in enumerate(ORDERED_INSTRS):         # iterate same order for coherence
        if not chord2_pcs_cycle:                         # safety: if empty, skip
            continue                                     # nothing to voice
        pc = chord2_pcs_cycle[(idx + 1) % len(chord2_pcs_cycle)]  # offset rotation for variety
        # try nudging above the previous assigned pitch to avoid static repetition
        prev_pitch = per_instr_events[instr]["pitch"][-1] if per_instr_events[instr]["time"] and \
                     per_instr_events[instr]["time"][-1] == bar_off + hit_positions[0] else tess_center[instr]
        approx = clamp(prev_pitch + 2, INSTRUMENTS[instr]["range"][0], INSTRUMENTS[instr]["range"][1])  # slight lift
        pitch = nearest_pitch_for_pc(pc, instr, approx)  # snap to nearest in-range note with chosen pc
        per_instr_events[instr]["time"].append(bar_off + hit_positions[1])    # onset at the second hit
        per_instr_events[instr]["duration"].append(hit_duration)              # sustain similar ring
        per_instr_events[instr]["pitch"].append(int(pitch))                   # realized MIDI pitch
        per_instr_events[instr]["velocity"].append(VEL_TUTTI_END)             # loud tutti dynamic


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
        rng = INSTRUMENTS[instr]["tess"];  tess = INSTRUMENTS[instr]["tess"]
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

# ---------------- Bar 2 chord voicing ----------------

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
    base = {"double_bass": -12, "cello": -7, "bass_clarinet": -3, "trumpet": +1, "violin": +3, "alto_flute": +4}.get(instr, 0)
    if voicing_hint == "spread":
        base = int(base * 1)
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

def prompt_4x3over4_from_bar1_contract() -> Tuple[str, str]:
    system = """
    You output ONLY strict JSON. No prose.

    Goal: Compose FOUR consecutive bars of 3/4 (12 ticks each; total 48 ticks) for SIX monophonic instruments,
    AND define a Lutosławski-style AGGREGATE (register-banded). This is the SAME schema and rules as Bar 1,
    but repeated across four separate 3/4 bars.

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
    "bars": [
        {
        "rhythm": {
            "<instr>": {"time":[ints 0..11 strictly inc], "duration":[>0, same len]}, ... all six ...
        },
        "melody": {
            "<instr>": {"start_hint":"low|mid|high", "steps":[len(time)-1 ints]}, ... all six ...
        }
        },
        { ... }, { ... }, { ... }  // exactly 4 bar objects
    ],
    "meta": { "bands_count": int, "band_lengths": [ints], "union_size": int }
    }

    AGGREGATE HARD CONSTRAINTS:
    - pcs are ABSOLUTE pitch-classes (C=0..B=11).
    - Each band's pcs length is EXACTLY 3 or 4.
    - Bands’ pcs are pairwise DISJOINT; the UNION across all bands is EXACTLY 12.
    - ORDER of pcs within each band matters (used for degree stepping).
    - Avoid pure equal-step bands (prefer none).

    Per-bar material constraints (apply IN EACH of the four bars):
    - Instruments: alto_flute, violin, bass_clarinet, trumpet, cello, double_bass (all six appear).
    - Monophony per instrument: times strictly increasing; time[i] + duration[i] <= 12.
    - len(steps) == max(0, len(time)-1) for each instrument.

    Validation checklist (you MUST satisfy before returning JSON):
    1) Aggregate passes all hard rules.
    2) In EVERY bar and for EACH instrument: arrays align, times strictly increasing, durations positive and within bar, steps length matches.
    If any check fails, regenerate internally and return only a valid JSON object.
    """
    user = """
    Instruments: alto_flute, violin, bass_clarinet, trumpet, cello, double_bass.
    Soft target onset counts per bar (guideline): AF 6..10, Vn 5..9, BCl 3..6, Tpt 2..5, Vc 2..5, Db 2..4.
    Return VALID JSON ONLY (no comments, no prose).
    """
    return system.strip(), user.strip()

# ---------------- Main Program ----------------
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
        rng = INSTRUMENTS[instr]["tess"];  tess = INSTRUMENTS[instr]["tess"]
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
        rng = INSTRUMENTS[instr]["tess"];  tess = INSTRUMENTS[instr]["tess"]
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

    # ---------------- Contract G: Bars 21–34 long-drone phrase ----------------

    _ALLOWED_PC_INTERVALS = {3, 4, 5, 7, 8, 9}  # tertian (3/4) + quartal (5) and their inversions

    def prompt_bars2134_drones_contract() -> Tuple[str, str]:
        """
        Bars 21–34: fourteen bars of 4/4 (16 ticks), slow long notes, max 4 players together,
        no simultaneous onsets (no vertical chord attacks), never complete silence (anchor each bar).
        LLM returns concrete MIDI pitches (range-legal) and times/durations per bar.
        """
        system = """
    You output ONLY strict JSON. No prose.

    Return EXACTLY:
    {
    "style": "long-drones",
    "bars": [
        {
        "anchor": "alto_flute|violin|bass_clarinet|trumpet|cello|double_bass",
        "events": [
            {"instrument":"<name>","time":int 0..15,"duration":int 8..16,"pitch":int MIDI},
            ... // 2..4 events total in this bar, unique instruments
        ]
        },
        ... // exactly 14 bar objects
    ]
    }

    Hard constraints you MUST satisfy BEFORE returning JSON:
    - Each bar has 2..4 events (unique instruments) and includes the anchor instrument.
    - Anchor event: time==0 AND duration==16 (fills the bar) → ensures no silence in this section.
    - No two events in the SAME bar may share the same 'time' (no chord onsets).
    - Max 4 players overlap at any moment (2..4 events already helps; keep durations long).
    - Durations are long-only: 8..16 ticks.
    - Pitches are absolute MIDI integers and MUST be within each instrument’s sounding range provided.
    - Harmony: whenever two instruments overlap in time, the interval between their PITCH CLASSES (mod 12)
    must be tertian or quartal: allowed semitone distances {3,4,5,7,8,9}.

    Validation: If any rule would be violated, regenerate internally. Return only a valid JSON object.
    """
        ranges = {k: INSTRUMENTS[k]["range"] for k in ORDERED_INSTRS}
        user = (
            "Meters are hard-coded: 14 bars of 4/4 (16 ticks per bar). "
            "Instrument ranges (sounding MIDI): " + json.dumps(ranges) + "\n"
            "Design a gradually evolving wave: rotate anchors across bars, vary which 2..4 players sustain; "
            "stagger onsets (no simult. times), and prefer smooth registral choreography."
        )
        return system.strip(), user.strip()

    def _events_overlap(e1, e2) -> bool:
        a0, a1 = int(e1["time"]), int(e1["time"]) + int(e1["duration"])
        b0, b1 = int(e2["time"]), int(e2["time"]) + int(e2["duration"])
        return max(a0, b0) < min(a1, b1)  # strict overlap in [start, end)

    def _pc_interval_ok(p1: int, p2: int) -> bool:
        d = abs(int(p1) - int(p2)) % 12
        return d in _ALLOWED_PC_INTERVALS

    def _check_bar_concurrency(events: List[dict]) -> Tuple[bool, str]:
        # Count active voices at each tick [0..15]
        active = [0]*16
        for ev in events:
            t0 = int(ev["time"]); t1 = min(16, int(ev["time"]) + int(ev["duration"]))
            for k in range(max(0,t0), max(0,t1)):
                if 0 <= k < 16:
                    active[k] += 1
        if any(c == 0 for c in active):
            return False, "bar has silent ticks (violates 'never a break' rule)"
        if any(c > 4 for c in active):
            return False, "bar exceeds max 4 concurrent players"
        return True, ""

    def validate_bars2134_or_raise(plan: dict) -> None:
        if not isinstance(plan, dict) or plan.get("style") != "long-drones":
            raise RuntimeError("style must be 'long-drones'")
        bars = plan.get("bars")
        if not (isinstance(bars, list) and len(bars) == 14):
            raise RuntimeError("bars must be a list of exactly 14 objects")

        valid_names = set(ORDERED_INSTRS)
        rngs = {k: INSTRUMENTS[k]["range"] for k in ORDERED_INSTRS}

        for bi, bar in enumerate(bars):
            anchor = bar.get("anchor")
            events = bar.get("events")
            if anchor not in valid_names:
                raise RuntimeError(f"bar {21+bi}: anchor must be a valid instrument")
            if not (isinstance(events, list) and 2 <= len(events) <= 4):
                raise RuntimeError(f"bar {21+bi}: events must be 2..4")

            names = [e.get("instrument") for e in events if isinstance(e, dict)]
            if len(names) != len(set(names)):
                raise RuntimeError(f"bar {21+bi}: duplicate instruments in events")
            if anchor not in names:
                raise RuntimeError(f"bar {21+bi}: anchor instrument missing from events")

            # field checks + range + time uniqueness + anchor geometry
            times = []
            for ev in events:
                instr = ev.get("instrument")
                if instr not in valid_names:
                    raise RuntimeError(f"bar {21+bi}: invalid instrument in events")
                t = ev.get("time"); d = ev.get("duration"); p = ev.get("pitch")
                if not (isinstance(t, int) and 0 <= t <= 15):
                    raise RuntimeError(f"bar {21+bi} {instr}: time must be 0..15")
                if not (isinstance(d, int) and 8 <= d <= 16):
                    raise RuntimeError(f"bar {21+bi} {instr}: duration must be 8..16")
                lo, hi = INSTRUMENTS[instr]["range"]
                if not (isinstance(p, int) and lo <= p <= hi):
                    raise RuntimeError(f"bar {21+bi} {instr}: pitch {p} out of range {INSTRUMENTS[instr]['range']}")
                times.append(t)

            if len(times) != len(set(times)):
                raise RuntimeError(f"bar {21+bi}: onsets must be time-unique (no chord onsets)")

            # anchor must be full-bar at t=0
            anchor_ev = [e for e in events if e["instrument"] == anchor][0]
            if not (int(anchor_ev["time"]) == 0 and int(anchor_ev["duration"]) == 16):
                raise RuntimeError(f"bar {21+bi}: anchor must start at 0 and last 16 ticks")

            # concurrency and continuity
            ok, msg = _check_bar_concurrency(events)
            if not ok:
                raise RuntimeError(f"bar {21+bi}: {msg}")

            # quartal/tertian overlap checks
            for i in range(len(events)):
                for j in range(i+1, len(events)):
                    e1, e2 = events[i], events[j]
                    if _events_overlap(e1, e2):
                        if not _pc_interval_ok(e1["pitch"], e2["pitch"]):
                            raise RuntimeError(
                                f"bar {21+bi}: overlapping {e1['instrument']}–{e2['instrument']} not quartal/tertian"
                            )

    def request_bars2134_drones_strict(client: OpenAI, attempts: int = 6, temp: float = 0.6) -> dict:
        last_err = None
        for _ in range(max(1, attempts)):
            sysG, usrG = prompt_bars2134_drones_contract()
            plan = call_llm_json(client=client, system_msg=sysG, user_msg=usrG,
                                model="gpt-4.1", temperature=temp, top_p=0.9)
            try:
                validate_bars2134_or_raise(plan)
                return plan
            except Exception as e:
                msg = str(e)
                # Relaxed policy: allow non-quartal/non-tertian overlaps, but keep all other checks strict.
                if "not quartal/tertian" in msg:
                    print("[warn] Contract G: harmony rule relaxed; accepting plan:", msg)
                    return plan
                last_err = msg
                continue
        raise RuntimeError("Bars 21–34 long-drone plan failed validation:\n" + (last_err or "(no details)"))
    
    def realize_4x3over4(plan: dict,
                     validated_agg: dict,
                     bar_offsets: List[int],
                     ticks_per_bar: int = 12) -> Dict[str, Dict[str, List[int]]]:
        """
        Realize a 4×3/4 fragment returned by prompt_4x3over4_from_bar1_contract().

        Inputs
        - plan: JSON with keys:
            {
                "aggregate_scale": {...},        # already validated externally
                "bars": [
                { "rhythm": { "<instr>": {"time":[...], "duration":[...]} , ... },
                    "melody": { "<instr>": {"start_hint":"low|mid|high","steps":[...]} , ... }
                },
                ... (4 bars total)
                ]
            }
        - validated_agg: aggregate dict already run through agg_validate_prepare_or_repair
        - bar_offsets: absolute start ticks for the 4 bars (len == 4)
        - ticks_per_bar: expected to be 12 (3/4 at 1/16 tick), but kept parametric

        Output
        - { "<instr>": {"time":[], "duration":[], "pitch":[], "velocity":[]}, ... }
            (velocity is set to a quiet default; you already override it at append-time)
        """
        if not (isinstance(plan, dict) and isinstance(plan.get("bars"), list) and len(plan["bars"]) == 4):
            raise RuntimeError("realize_4x3over4: plan.bars must be a list of 4 bar objects")
        if not (isinstance(bar_offsets, list) and len(bar_offsets) == 4):
            raise RuntimeError("realize_4x3over4: bar_offsets must be a list of 4 absolute ticks")

        out = {instr: {"time": [], "duration": [], "pitch": [], "velocity": []}
            for instr in ORDERED_INSTRS}

        agg = validated_agg  # already validated outside

        for bi in range(4):
            bar = plan["bars"][bi] or {}
            rhythm = (bar.get("rhythm") or {})
            melody = (bar.get("melody") or {})
            off = int(bar_offsets[bi])

            for instr in ORDERED_INSTRS:
                if instr not in rhythm:
                    # Keep strict: if the plan omitted an instrument, fail fast so we can fix upstream.
                    raise RuntimeError(f"4×3/4 bar {bi} rhythm missing instrument '{instr}'")

                # --- Clean local time/duration
                t_local = [int(x) for x in (rhythm[instr].get("time") or [])]
                d_local = [int(x) for x in (rhythm[instr].get("duration") or [])]

                t_local = enforce_monophony_times([x for x in t_local if 0 <= x < ticks_per_bar])
                t_local, d_local = cap_durations_local(t_local, d_local, ticks_per_bar)

                # --- Melody → pitches via aggregate degree-walk (fallback to scale-walk if steps mismatch)
                rng  = INSTRUMENTS[instr]["tess"]
                tess = INSTRUMENTS[instr]["tess"]
                mdef = (melody.get(instr) or {})
                steps = mdef.get("steps", [])
                start_hint = mdef.get("start_hint", "mid")

                if isinstance(steps, list) and len(steps) == max(0, len(t_local) - 1):
                    pitches = pitches_from_steps_aggregate(t_local, steps, agg, rng, tess, start_hint)
                else:
                    pitches = scale_walk_assign_pitches_aggregate(t_local, agg, rng, tess)

                # --- Absolute placement
                t_abs = absolute_times(t_local, off)

                # Default quiet velocity (you override when appending; harmless to include here)
                vels = [VEL_BAR3_PP for _ in t_abs]

                # Append
                out[instr]["time"]     += t_abs
                out[instr]["duration"] += d_local
                out[instr]["pitch"]    += pitches
                out[instr]["velocity"] += vels

        return out


    def realize_bars2134_drones(plan: dict) -> Dict[str, Dict[str, List[int]]]:
        """Realize Bars 21–34 from the validated plan. (Bar 20 is a silent short break.)"""
        out = {instr: {"time": [], "duration": [], "pitch": [], "velocity": []} for instr in ORDERED_INSTRS}
        bars = plan.get("bars", []) or []
        # Offsets for bars 21..34
        offsets = [BAR21_OFF + 16*i for i in range(14)]
        for bi, bar in enumerate(bars[:14]):
            off = offsets[bi]
            for ev in sorted(bar.get("events", []), key=lambda e: int(e["time"])):
                instr = ev["instrument"]
                t = int(ev["time"]); d = int(ev["duration"]); p = int(ev["pitch"])
                tlo, thi = INSTRUMENTS[instr]["tess"]
                p = clamp(p, tlo, thi)
                out[instr]["time"].append(off + t)
                out[instr]["duration"].append(d)
                out[instr]["pitch"].append(p)
                out[instr]["velocity"].append(VEL_DRONE)
        return out
    
    def enrich_bars2134_chords(per_instr_events: Dict[str, Dict[str, List[int]]],
                           seed: int = None,
                           density: float = 0.55,
                           max_extra_per_bar: int = 2,
                           max_new_per_onset: int = 1) -> None:
        """
        Post-process bars 21–34: at some onsets, add short notes in currently-silent instruments
        to create verticals. Keeps ≤4 concurrent players, shortens if needed to fit the limit.
        Mutates per_instr_events in place.
        """
        rng = random.Random(seed)
        allowed_pc_steps = (3, 4, 5, 7, 8, 9)

        def bar_bounds(bi: int) -> Tuple[int, int]:
            start = BAR21_OFF + 16 * bi
            return start, start + 16

        def active_at(t_abs: int) -> List[Tuple[str, int]]:
            """(instr, pitch) currently sounding at absolute tick t_abs."""
            out = []
            for instr in ORDERED_INSTRS:
                T = per_instr_events[instr]["time"]
                D = per_instr_events[instr]["duration"]
                P = per_instr_events[instr]["pitch"]
                for i in range(len(T)):
                    if T[i] <= t_abs < T[i] + D[i]:
                        out.append((instr, int(P[i])))
            return out

        def build_bar_snapshot(bi: int):
            start, end = bar_bounds(bi)
            # events starting in this bar
            onsets = {}   # local_t -> (instr, pitch, dur)
            present_instrs = set()
            # occupancy per local tick (16 long)
            occ = [0] * 16
            sounding = []  # (instr, local_t, dur, pitch)
            for instr in ORDERED_INSTRS:
                T = per_instr_events[instr]["time"]
                D = per_instr_events[instr]["duration"]
                P = per_instr_events[instr]["pitch"]
                for i in range(len(T)):
                    t, d, p = int(T[i]), int(D[i]), int(P[i])
                    if start <= t < end:
                        lt = t - start
                        onsets.setdefault(lt, (instr, p, d))
                        present_instrs.add(instr)
                        sounding.append((instr, lt, d, p))
                    # fill occupancy if any part lies in this bar
                    s0 = max(t, start); s1 = min(t + d, end)
                    if s0 < s1:
                        for k in range(s0 - start, s1 - start):
                            occ[k] += 1
            return onsets, present_instrs, occ

        def choose_pitch_relative(base_pitch: int, instr: str) -> int:
            """Pick nearest pitch in instr range whose pc forms a tertian/quartal interval with base."""
            lo, hi = INSTRUMENTS[instr]["range"]
            tess = INSTRUMENTS[instr]["tess"]
            approx_center = _spread_bias(instr, _center_from_register_hint(tess, "mid"), "spread")
            base_pc = base_pitch % 12
            pcs = []
            for d in allowed_pc_steps:
                pcs.append((base_pc + d) % 12)
                pcs.append((base_pc - d) % 12)
            # stable but varied: small shuffle
            rng.shuffle(pcs)
            # try to avoid exact pitch dupes at the onset
            return _nearest_pitch_for_pcset(pcs[0], approx_center, (lo, hi))

        # iterate bars 21..34 (index 0..13)
        for bi in range(14):
            start, end = bar_bounds(bi)
            onsets, present_instrs, occ = build_bar_snapshot(bi)
            if not onsets:
                continue

            # candidate onsets: sort; we mildly prefer non-zero times to create motion
            times_sorted = sorted(onsets.keys(), key=lambda t: (t == 0, t))
            added_this_bar = 0

            for lt in times_sorted:
                if added_this_bar >= max_extra_per_bar:
                    break
                # density gate
                if rng.random() > density:
                    continue

                abs_t = start + lt
                currently_sounding = active_at(abs_t)  # [(instr, pitch), ...]
                conc_now = len(currently_sounding)
                if conc_now >= 4:
                    continue

                # reference pitch: the note that *starts* here if any, else the lowest sounding
                base_instr, base_pitch, _ = onsets.get(lt, (None, None, None))
                if base_pitch is None:
                    if not currently_sounding:
                        continue
                    base_pitch = min(p for _, p in currently_sounding)

                # eligible instruments: not already present in this bar AND not sounding now
                busy_now = {i for i, _ in currently_sounding}
                eligible = [i for i in ORDERED_INSTRS if i not in present_instrs and i not in busy_now]

                if not eligible:
                    continue

                can_add = min(max_new_per_onset, 4 - conc_now)
                picks = eligible[:can_add]  # deterministic order

                for instr in picks:
                    # find max duration s.t. occupancy never exceeds 4
                    max_len = min(8, 16 - lt)  # short-ish pulses
                    if max_len <= 0:
                        continue
                    # proposed pitch
                    pitch = choose_pitch_relative(base_pitch, instr)

                    # shrink to fit concurrency limit
                    dur = 0
                    for k in range(max_len):
                        if occ[lt + k] >= 4:
                            break
                        dur += 1
                    if dur < 3:
                        continue  # too short to be useful

                    # apply
                    per_instr_events[instr]["time"].append(abs_t)
                    per_instr_events[instr]["duration"].append(dur)
                    per_instr_events[instr]["pitch"].append(int(pitch))
                    per_instr_events[instr]["velocity"].append(VEL_DRONE_HARM)

                    # update occupancy and book-keeping
                    for k in range(dur):
                        occ[lt + k] += 1
                    present_instrs.add(instr)
                    added_this_bar += 1
                    if added_this_bar >= max_extra_per_bar:
                        break
    
    def overlay_bars2134_ripples(
        per_instr_events: Dict[str, Dict[str, List[int]]],  # global per-instrument accumulator (mutated)
        seed: int = None,                                   # random seed for determinism (set to an int to freeze)
        prob_per_bar: float = 1.0,                          # chance to activate *this bar* for ripples
        max_ripples_per_bar: int = 3,                       # NEW: place up to N separate ripples per bar
        concurrency_cap: int = 6,                           # NEW: allow more simultaneous voices (<= cap)
        min_notes: int = 3,                                 # ripple length lower bound (notes)
        max_notes: int = 6,                                 # ripple length upper bound (notes)
        note_len_choices: Tuple[int, ...] = (1, 2),         # per-note duration choices (ticks)
        gap_choices: Tuple[int, ...] = (1, 2),              # gap between notes (ticks)
        avoid_instruments_with_onsets_in_bar: bool = False  # NEW: if True, skip instruments that *start* in this bar
    ) -> None:
        """
        Bars 21–34: sprinkle short, bright, monophonic “ripple” runs.
        Upgrades:
        - Multiple ripples per bar (max_ripples_per_bar).
        - Adjustable concurrency limit (concurrency_cap).
        - Optional relaxation of 'present onset in bar' exclusion.
        Never places a ripple onset on a tick whose occupancy already reaches concurrency_cap.
        """

        rng = random.Random(seed)                           # deterministic RNG if seed is given
        allowed_pc_steps = (3, 4, 5, 7, 8, 9)               # tertian/quartal distances in semitones

        def bar_bounds(bi: int) -> Tuple[int, int]:
            """Absolute [start,end) tick window for Bar (21+bi)."""
            start = BAR21_OFF + 16 * bi                     # fixed 16-tick bars here
            return start, start + 16

        def active_at(t_abs: int) -> List[Tuple[str, int]]:
            """Return list of (instrument, pitch) sounding at absolute tick t_abs."""
            out = []                                        # accumulator
            for instr in ORDERED_INSTRS:                    # scan all instruments
                T = per_instr_events[instr]["time"]         # onsets (abs)
                D = per_instr_events[instr]["duration"]     # durations (ticks)
                P = per_instr_events[instr]["pitch"]        # pitches (MIDI)
                for i in range(len(T)):                     # check each event
                    if T[i] <= t_abs < T[i] + D[i]:         # half-open interval test
                        out.append((instr, int(P[i])))      # add currently sounding event
            return out                                      # snapshot at t_abs

        def build_bar_snapshot(bi: int):
            """Compute per-bar occupancy (local 0..15), map of onsets at local ticks, and set of instruments with onsets."""
            start, end = bar_bounds(bi)                     # bar window
            onsets = {}                                     # local tick -> (instr, pitch, dur)
            present_instrs = set()                          # instruments that *start* something in this bar
            occ = [0] * 16                                  # occupancy per local tick
            for instr in ORDERED_INSTRS:                    # across all instruments
                T = per_instr_events[instr]["time"]         # their onsets
                D = per_instr_events[instr]["duration"]     # their durations
                P = per_instr_events[instr]["pitch"]        # their pitches
                for i in range(len(T)):                     # scan events
                    t, d, p = int(T[i]), int(D[i]), int(P[i])  # normalize ints
                    if start <= t < end:                    # event starts inside this bar
                        lt = t - start                      # convert to local tick
                        onsets.setdefault(lt, (instr, p, d))# remember a representative starter
                        present_instrs.add(instr)           # mark instrument as 'present' (onset in this bar)
                    s0 = max(t, start)                      # overlap start with this bar
                    s1 = min(t + d, end)                    # overlap end
                    if s0 < s1:                             # if some overlap exists
                        for k in range(s0 - start, s1 - start):
                            occ[k] += 1                     # increment occupancy on covered ticks
            return onsets, present_instrs, occ              # return snapshot structures

        def _nearest_pitch_for_pcset(target_pc: int, approx: int, rng_midi: Tuple[int,int]) -> int:
            """Nearest in-range pitch with pitch-class target_pc, centered around approx."""
            lo, hi = rng_midi                               # sounding range for the instrument
            best, bestd = approx, 10**9                     # track nearest candidate
            for k in range(-24, 25):                        # search ±2 octaves
                cand = clamp(approx + k, lo, hi)            # clamp within range
                if cand % 12 == target_pc:                  # pc match?
                    d = abs(cand - approx)                  # distance to approx
                    if d < bestd:                           # keep nearest
                        best, bestd = cand, d
            return best                                     # snapped pitch

        def choose_pitch_relative(base_pitch: int, instr: str) -> int:
            """Pick starting pitch for instr on a tertian/quartal pc from base_pitch."""
            lo, hi = INSTRUMENTS[instr]["range"]            # absolute range
            tess = INSTRUMENTS[instr]["tess"]               # tessitura preference
            approx_center = (tess[0] + tess[1]) // 2        # mid tessitura
            base_pc = base_pitch % 12                       # base pitch-class
            pcs = []                                        # candidate pcs (± allowed steps)
            for d in allowed_pc_steps:                      # iterate allowed distances
                pcs.append((base_pc + d) % 12)              # upward
                pcs.append((base_pc - d) % 12)              # downward
            rng.shuffle(pcs)                                # small shuffle for variety
            return _nearest_pitch_for_pcset(pcs[0], approx_center, (lo, hi))  # snap to closest

        def next_ripple_pitch(prev_pitch: int, instr: str) -> int:
            """Continue ripple by a small tertian/quartal pc hop around previous pitch."""
            step = rng.choice(allowed_pc_steps)             # choose interval size
            direction = rng.choice((+1, -1))                # direction
            target_pc = (prev_pitch + direction * step) % 12# next pitch-class
            lo, hi = INSTRUMENTS[instr]["range"]            # instrument range
            approx = clamp(prev_pitch + direction * 2, lo, hi)  # bias toward movement direction
            return _nearest_pitch_for_pcset(target_pc, approx, (lo, hi))       # snap

        # ---- iterate bars 21..34 (bi=0..13) ----
        for bi in range(14):                                # 14 bars to process
            if rng.random() > prob_per_bar:                 # probabilistic gate for this bar
                continue                                    # skip bar → no ripples at all

            placed = 0                                      # number of ripples placed in this bar
            attempts = 0                                    # safety to avoid infinite loops
            # keep trying until we place enough ripples or run out of room
            while placed < max_ripples_per_bar and attempts < 12:
                attempts += 1                               # count this placement attempt

                start, end = bar_bounds(bi)                 # absolute bar window
                onsets, present_instrs, occ = build_bar_snapshot(bi)  # recompute snapshot after prior placements
                if sum(occ) == 0:                           # empty bar (unlikely) → nothing to do
                    break                                   # exit attempts for this bar

                # --- choose a start tick with spare capacity (< concurrency_cap) ---
                candidates = list(range(0, 16))             # allow 0..15 now for more chances
                rng.shuffle(candidates)                      # randomize scanning
                lt0 = None                                   # chosen local start tick
                for tloc in candidates:                      # scan local ticks
                    # check onset capacity at the tick
                    if occ[tloc] >= concurrency_cap:        # already at cap → not usable
                        continue                             # try next tick
                    lt0 = tloc                               # accept this start tick
                    break                                    # stop search once found
                if lt0 is None:                              # nowhere to start a ripple
                    break                                    # give up on this bar

                abs_t0 = start + lt0                         # absolute start time
                sounding_now = active_at(abs_t0)             # who is sounding at start?
                conc_now = len(sounding_now)                 # current occupancy
                if conc_now >= concurrency_cap:              # double-check (absolute snapshot)
                    continue                                 # try again with another start

                busy_now = {i for i, _ in sounding_now}      # instruments currently sounding
                # decide pool of eligible instruments
                if avoid_instruments_with_onsets_in_bar:     # optional stricter mode
                    eligible = [i for i in ORDERED_INSTRS if i not in busy_now and i not in present_instrs]
                else:                                        # relaxed: only require silence *at the onset*
                    eligible = [i for i in ORDERED_INSTRS if i not in busy_now]

                if not eligible:                             # nobody free at this tick
                    continue                                 # try a different start

                instr = eligible[0]                          # deterministic pick for stability

                # --- synthesize a short ripple gesture (respecting concurrency_cap per covered tick) ---
                notes_n = clamp(rng.randint(min_notes, max_notes), 1, 8)  # how many notes in gesture
                times_local: List[int] = []                  # local onset times we’ll place
                durs_local: List[int]  = []                  # durations for each onset
                cursor = lt0                                 # start cursor at chosen tick
                for _ in range(notes_n):                     # place up to notes_n short notes
                    if cursor >= 16:                         # ran out of bar
                        break                                # stop building
                    dur = rng.choice(note_len_choices)       # choose a short dur (1–2)
                    dur = min(dur, 16 - cursor)              # clamp within bar
                    # ensure every covered tick stays below cap
                    ok = all(occ[cursor + k] < concurrency_cap for k in range(dur))
                    if not ok:                               # if too full…
                        # try shrinking to 1-tick atom
                        dur = 1
                        ok = occ[cursor] < concurrency_cap
                    if not ok:                               # still no space at cursor
                        cursor += 1                          # nudge right and retry next cycle
                        continue
                    times_local.append(cursor)               # accept onset
                    durs_local.append(dur)                   # accept duration
                    for k in range(dur):                     # update occupancy model
                        occ[cursor + k] += 1                 # reserve capacity for these ticks
                    gap = rng.choice(gap_choices)            # choose small gap
                    cursor += max(1, gap)                    # advance cursor

                if not times_local:                          # nothing actually placed
                    continue                                 # try another start or bar

                # --- pick a reference pitch for harmonic flavor near this start ---
                base_instr, base_pitch, _ = onsets.get(lt0, (None, None, None))  # onset-aligned ref if present
                if base_pitch is None:                       # else: use current lowest pitch at abs_t0, fallback to tess center
                    cur = active_at(abs_t0)                  # snapshot
                    if cur:
                        base_pitch = min(p for _, p in cur)  # lowest sounding pitch now
                    else:
                        tlo, thi = INSTRUMENTS[instr]["tess"]# tessitura band
                        base_pitch = (tlo + thi) // 2        # center pitch

                # --- generate ripple pitches via small tertian/quartal hops ---
                first_pitch = choose_pitch_relative(base_pitch, instr)           # opening pitch near base
                pitches = [int(first_pitch)]                                     # seed
                for _ in range(1, len(times_local)):                             # continue gesture
                    pitches.append(int(next_ripple_pitch(pitches[-1], instr)))   # hop

                # --- append ripple to chosen instrument (absolute placement) ---
                abs_times = [start + t for t in times_local]                     # local → absolute
                for t_abs, dur, pitch in zip(abs_times, durs_local, pitches):    # write each note
                    per_instr_events[instr]["time"].append(t_abs)                # onset
                    per_instr_events[instr]["duration"].append(dur)              # duration
                    per_instr_events[instr]["pitch"].append(int(pitch))          # pitch
                    per_instr_events[instr]["velocity"].append(VEL_RIPPLE)       # bright dynamic

                placed += 1                                                      # count this ripple
            # end while
            # end for bars

    # ---- Short break (Bar 20: 1/4 silence), then Bars 21–34 long-drone phrase ----
    # (Break requires no events added; the meter map carries the bar.)
    plan2134 = request_bars2134_drones_strict(oai, attempts=8, temp=0.6)
    bars2134 = realize_bars2134_drones(plan2134)

    # Append Bars 21–34 to per-instrument accumulators
    for instr in ORDERED_INSTRS:
        part = bars2134[instr]
        per_instr_events[instr]["time"]    += part["time"]
        per_instr_events[instr]["duration"]+= part["duration"]
        per_instr_events[instr]["pitch"]   += part["pitch"]
        per_instr_events[instr]["velocity"]+= part["velocity"]

    # Add chordal movement in bars 21–34 by waking silent instruments on select onsets
    seedG_enrich = random.randint(1, 10**9)
    enrich_bars2134_chords(
        per_instr_events,
        seed=seedG_enrich,
        density=0.6,           # more/less frequent chords
        max_extra_per_bar=2,   # how many extra hits per bar
        max_new_per_onset=1    # how many instruments join per onset
    )

    # Add brief “Textural Ripples” over bars 21–34 (fast monophonic runs in silent voices)
    # DENser ripples over bars 21–34
    seedG_ripples = random.randint(1, 10**9)  # set to a constant for reproducible runs
    overlay_bars2134_ripples(
        per_instr_events,
        seed=seedG_ripples,
        prob_per_bar=0.5,            # try every bar
        max_ripples_per_bar=1,       # up to 1 gesture per bar
        concurrency_cap=6,           # allow one or two extra simultaneous voices
        min_notes=3, max_notes=6,    # short gestures
        note_len_choices=(1, 2),     # fast notes
        gap_choices=(1, 2),          # tight spacing
        avoid_instruments_with_onsets_in_bar=False  # relaxed eligibility for more options
    )

    # ---- Bars 35–37: LITERAL repeat of Bars 4–6 (absolute shift) ----
    # bars456 already realized earlier with absolute times at BAR4_OFF/BAR5_OFF/BAR6_OFF.
    SHIFT_456_TO_35 = BAR35_OFF - BAR4_OFF   # 464 - 36 = 428
    for instr in ORDERED_INSTRS:
        src = bars456[instr]
        per_instr_events[instr]["time"]     += [t + SHIFT_456_TO_35 for t in src["time"]]
        per_instr_events[instr]["duration"] += src["duration"][:]
        per_instr_events[instr]["pitch"]    += src["pitch"][:]
        per_instr_events[instr]["velocity"] += src["velocity"][:]

    # ---- Bars 38–40: VARIATION by re-running Contract C exactly the same way ----
    # Reuse the same prompts sysC/usrC and same request path; just include agg456 in the prior list so it's nudged to differ.
    pc456_var, agg456_var = request_fresh_aggregate(
        make_prompts_fn=mk_prompts_C,
        key_in_json="aggregate_scale",
        client=oai,
        prior_aggs=[b1_agg, contrast_agg] + extra_bar2_aggs + [agg456],
        attempts=4, temp=1.15
    )
    bars456_var = realize_bars456(pc456_var)

    SHIFT_456_TO_38 = BAR38_OFF - BAR4_OFF   # 508 - 36 = 472
    for instr in ORDERED_INSTRS:
        src = bars456_var[instr]
        per_instr_events[instr]["time"]     += [t + SHIFT_456_TO_38 for t in src["time"]]
        per_instr_events[instr]["duration"] += src["duration"][:]
        per_instr_events[instr]["pitch"]    += src["pitch"][:]
        per_instr_events[instr]["velocity"] += src["velocity"][:]

    # ---- Bar 41: short break (1/4) ----
    # Intentional silence; we do NOT append any events. Meter map will carry the bar.

    # ---- Bars 42–45: re-run Bar 1’s contract as a 4×3/4 fragment ----
    sysX, usrX = prompt_4x3over4_from_bar1_contract()
    seedX = random.randint(1, 10**9)
    usrX += f"\nCreativeSeed: {seedX}\nRule: When multiple valid choices exist, bias decisions with CreativeSeed (aggregate & rhythms)."

    bars4x, = (call_llm_json(oai, sysX, usrX, model="gpt-4.1", temperature=0.8),)

    # we assume your first fragment plan JSON object is named `bars4x` (the LLM plan, not the realized notes)
    prev_plan_min = _compress_plan(bars4x)

    # Validate/prepare aggregate for this 4×3/4 fragment
    if "aggregate_scale" not in bars4x or "bars" not in bars4x or not isinstance(bars4x["bars"], list) or len(bars4x["bars"]) != 4:
        raise RuntimeError("4×3/4 JSON missing aggregate_scale or 'bars' (len != 4).")
    agg4x = agg_validate_prepare_or_repair(bars4x["aggregate_scale"])

    # Realize each of the four 3/4 bars:
    # - ALL instruments EXCEPT trumpet: play the generated material, but at ppp (very silent).
    # - TRUMPET: ignore generated material; instead, 3 long loud notes (Bars 42–44) derived from agg4x.
    four_offsets = [BAR42_OFF, BAR43_OFF, BAR44_OFF, BAR45_OFF]
    non_trumpet = [i for i in ORDERED_INSTRS if i != "trumpet"]

    for bi in range(4):
        bar_payload = bars4x["bars"][bi]
        rhythmX: Dict[str,dict] = bar_payload.get("rhythm", {}) or {}
        melodyX: Dict[str,dict] = bar_payload.get("melody", {}) or {}

        # --- Non-trumpet instruments: realize at ppp
        for instr in non_trumpet:
            if instr not in rhythmX:
                raise RuntimeError(f"4×3/4 bar {42+bi}: rhythm missing {instr}")
            t_local = rhythmX[instr].get("time", [])
            d_local = rhythmX[instr].get("duration", [])

            # Enforce local constraints (3/4 → 12 ticks)
            t_local = enforce_monophony_times([int(x) for x in t_local if 0 <= int(x) < 12])
            t_local, d_local = cap_durations_local(t_local, [int(x) for x in d_local], 12)
            rhythmX[instr]["time"] = t_local
            rhythmX[instr]["duration"] = d_local

            # Pitches from degree-steps under the NEW aggregate (same logic as Bar 1)
            rng = INSTRUMENTS[instr]["tess"];  tess = INSTRUMENTS[instr]["tess"]
            m = (melodyX.get(instr) or {})
            steps = m.get("steps", [])
            start_hint = m.get("start_hint", "mid")
            if isinstance(steps, list) and len(steps) == max(0, len(t_local) - 1):
                pitches = pitches_from_steps_aggregate(t_local, steps, agg4x, rng, tess, start_hint)
            else:
                pitches = scale_walk_assign_pitches_aggregate(t_local, agg4x, rng, tess)

            # Super quiet: ppp
            vels = [VEL_PPP for _ in t_local]

            # Append with absolute offsets for Bars 42–45
            t_abs = absolute_times(t_local, four_offsets[bi])
            per_instr_events[instr]["time"]     += t_abs
            per_instr_events[instr]["duration"] += d_local
            per_instr_events[instr]["pitch"]    += pitches
            per_instr_events[instr]["velocity"] += vels

    # --- TRUMPET OVERRIDE (Bars 42–44): three long loud notes, LLM-chosen from agg4x ---
    seedT_A = random.randint(1, 10**9)
    tpt_pitches = choose_trumpet_long_notes_via_llm(oai, agg4x, seedT_A, "trumpet", avoid_midis=None)

    for t, d, p in zip([BAR42_OFF, BAR43_OFF, BAR44_OFF], [12, 12, 12], tpt_pitches):
        per_instr_events["trumpet"]["time"].append(t)
        per_instr_events["trumpet"]["duration"].append(d)
        per_instr_events["trumpet"]["pitch"].append(int(p))
        per_instr_events["trumpet"]["velocity"].append(VEL_TRUMPET_LOUD)

    # ---- Bars 46–49: responding 4×3/4 fragment, different harmony; all ppp except trumpet continues long loud notes ----
    # Reuse the same 4×3/4 prompt helper as for Bars 42–45, but ask for a materially different aggregate.
    sysY, usrY = prompt_4x3over4_from_bar1_contract()
    seedY = random.randint(1, 10**9)
    # Encourage a "response" aggregate: change pc→band mapping vs previous agg4x
    prev_map_4x = pc_to_band_map(agg4x)

    usrY += (
        f"\nCreativeSeed: {seedY}"
        "\nDirective: Treat this as a 'response' to the previous 4×3/4 fragment."
        "\nHARD: Provide a materially different aggregate vs the previous fragment: "
        "change the pc→band assignment for at least 6 pitch classes and adjust band ranges."
        "\nReference pc→band map of previous fragment: " + json.dumps(prev_map_4x) +
        "\nHARD: RHYTHM VARIATION: For EACH non-trumpet instrument in EACH bar, "
        "the onset set must differ materially from the first fragment. "
        "Target Jaccard(onset_set_new, onset_set_old) < 0.5 per bar. "
        "Use at least one different duration value in every instrument/bar."
        "\nHARD: MELODIC VARIATION: For EACH non-trumpet instrument, provided 'steps' "
        "must NOT be an exact copy—alter at least half of the positions."
        "\nReferenceFirstFragment (to differ from): " + json.dumps(prev_plan_min)
    )

    bars4x_resp, = (call_llm_json(oai, sysY, usrY, model='gpt-4.1', temperature=0.8),)

    # Realize response fragment (Bars 46–49) from its aggregate
    agg4x_resp = agg_validate_prepare_or_repair(bars4x_resp["aggregate_scale"])
    events4x_resp = realize_4x3over4(bars4x_resp, agg4x_resp,
                                    bar_offsets=[BAR46_OFF, BAR47_OFF, BAR48_OFF, BAR49_OFF],
                                    ticks_per_bar=12)

    # Append to per-instrument streams (ppp), EXCEPT trumpet (overridden below)
    for instr in ORDERED_INSTRS:
        if instr == "trumpet":
            continue
        part = events4x_resp[instr]
        per_instr_events[instr]["time"]     += part["time"]
        per_instr_events[instr]["duration"] += part["duration"]
        per_instr_events[instr]["pitch"]    += part["pitch"]
        per_instr_events[instr]["velocity"] += [VEL_BAR3_PP] * len(part["time"])  # ppp

    # --- Trumpet (Bars 46–48): three long, loud notes again, but LLM-chosen AND different from Bars 42–44 ---
    seedT_B = random.randint(1, 10**9)
    tpt_pitches_R = choose_trumpet_long_notes_via_llm(oai, agg4x_resp, seedT_B, "trumpet", avoid_midis=tpt_pitches)

    for t, d, p in zip([BAR46_OFF, BAR47_OFF, BAR48_OFF], [12, 12, 12], tpt_pitches_R):
        per_instr_events["trumpet"]["time"].append(t)
        per_instr_events["trumpet"]["duration"].append(d)
        per_instr_events["trumpet"]["pitch"].append(int(p))
        per_instr_events["trumpet"]["velocity"].append(VEL_TRUMPET_LOUD)

    # === Part 1 → Final bar: two tutti jazz chords chosen by the LLM ===
    # 1) Look at the recent harmony (last 8 bars) to give the LLM context
    recent_pcs = collect_recent_pcs(per_instr_events, lookback_bars=8, bar_size=16)  # e.g., PCs used in Bars 27–34

    # 2) Ask the LLM for two complex jazz chords that avoid simple V→I and keep things suspended
    js_two = request_part1_finalbar_twochords_jazz(client, recent_pcs, temp=0.95)    # a touch more exploratory

    # 3) Validate/sanitize the chords (or fall back to safe suspended sets)
    chord1_pcs, chord2_pcs = validate_twochords_jazz_payload(js_two)                 # returns two pc arrays (5..8 pcs)

    # 4) Place the final bar at the next clean bar boundary and realize the two tutti hits
    final_bar_off = BAR50_OFF
    realize_part1_finalbar_tutti(                                                     
        per_instr_events=per_instr_events,                                           # global accumulator
        chord1_pcs=chord1_pcs,                                                       # first suspended chord
        chord2_pcs=chord2_pcs,                                                       # second suspended chord (related, unresolved)
        bar_off=final_bar_off,                                                       # final bar start
        bar_size=16,                                                                 # 4/4 → 16 ticks
        hit_positions=(0, 8),                                                        # hits on beats 1 and 3
        hit_duration=6                                                               # each rings for 6 ticks (2 ticks of air)
    )

    # (Optional) If you maintain a separate meter map structure for your visualiser,
    # append a 4/4 entry for this bar here, e.g.:
    # meter_map.append({"abs_off": final_bar_off, "numerator": 4, "denominator": 4})


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

            per_instr_events[instr]["time"]    += [GLOBAL_END - 1]*pad_n 
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
            {"meter":[1,4],"start":232},  # 19 recap
            {"meter":[1,4],"start":236},  # 20 short break (silence)
            {"meter":[4,4],"start":240},  # 21
            {"meter":[4,4],"start":256},  # 22
            {"meter":[4,4],"start":272},  # 23
            {"meter":[4,4],"start":288},  # 24
            {"meter":[4,4],"start":304},  # 25
            {"meter":[4,4],"start":320},  # 26
            {"meter":[4,4],"start":336},  # 27
            {"meter":[4,4],"start":352},  # 28
            {"meter":[4,4],"start":368},  # 29
            {"meter":[4,4],"start":384},  # 30
            {"meter":[4,4],"start":400},  # 31
            {"meter":[4,4],"start":416},  # 32
            {"meter":[4,4],"start":432},  # 33
            {"meter":[4,4],"start":448},   # 34
            {"meter":[3,4],"start":464},  # 35 (repeat of 4)
            {"meter":[4,4],"start":476},  # 36 (repeat of 5)
            {"meter":[4,4],"start":492},  # 37 (repeat of 6)
            {"meter":[3,4],"start":508},  # 38 (variation of 4)
            {"meter":[4,4],"start":520},  # 39 (variation of 5)
            {"meter":[4,4],"start":536},  # 40 (variation of 6)
            {"meter":[1,4],"start":552},  # 41 short break
            {"meter":[3,4],"start":556},  # 42
            {"meter":[3,4],"start":568},  # 43
            {"meter":[3,4],"start":580},  # 44
            {"meter":[3,4],"start":592},  # 45
            {"meter":[3,4],"start": 604},
            {"meter":[3,4],"start": 616},
            {"meter":[3,4],"start": 628},
            {"meter":[3,4],"start": 640},
        ],
        "N_global": N_global,
        "top_feature": top_name,
        "instrument_features": instrument_feature_names,
        "instrument_meta": INSTRUMENT_META,
        "tracks": tracks,
    }

    with open("program1_payload.json","w",encoding="utf-8") as f:
        json.dump(payload_out, f, ensure_ascii=False, indent=2)
    print("✓ Wrote program1_payload.json")
    print(f"Top-level DCN feature: {top_name}")
    print(f"N_global: {N_global}")

if __name__ == "__main__":
    main()