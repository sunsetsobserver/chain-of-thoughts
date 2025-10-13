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

def _ordered_scale_pcs(tonic_midi: int, pcs: List[int]) -> List[int]:
    """Return ordered pitch-classes of the scale anchored at tonic%12."""
    root = tonic_midi % 12
    return [(root + p) % 12 for p in pcs]

def _nearest_scale_member(m: int, scl_pcs: List[int]) -> int:
    """Snap MIDI 'm' to nearest pitch whose PC is in 'scl_pcs' (tie → prefer upward)."""
    pcset = set(scl_pcs)
    if (m % 12) in pcset:
        return m
    for d in range(1, 13):
        up = m + d
        dn = m - d
        if (up % 12) in pcset:
            return up
        if (dn % 12) in pcset:
            return dn
    return m

def _move_in_scale_degrees(current: int, steps: int, scl_pcs: List[int], rng: Tuple[int, int]) -> int:
    """
    Move by 'steps' SCALE-DEGREE steps (±) over an arbitrary ordered pitch-class set.
    We locate the nearest scale degree to 'current', shift by 'steps', wrap with octave
    adjustment, and clamp to 'rng'.
    """
    lo, hi = rng
    # anchor current to nearest scale pitch
    anchored = _nearest_scale_member(current, scl_pcs)
    cur_pc = anchored % 12
    degs = len(scl_pcs)
    # index of current degree (closest circularly)
    cur_idx = min(range(degs), key=lambda i: (scl_pcs[i] - cur_pc) % 12)
    new_idx = cur_idx + int(steps)
    oct_shift, rel = divmod(new_idx, degs)  # works for negatives too
    # keep octave near anchored pitch
    base_oct = anchored - (anchored % 12)
    target_pc = scl_pcs[rel]
    cand = base_oct + target_pc + 12 * oct_shift
    # pull back into range by octave if needed
    if cand < lo:
        k = (lo - cand + 11) // 12
        cand += 12 * k
    elif cand > hi:
        k = (cand - hi + 11) // 12
        cand -= 12 * k
    return clamp(cand, lo, hi)

def pitches_from_steps(times: List[int], steps: List[int], tonic_midi: int,
                       pcs: List[int], rng: Tuple[int, int], tess: Tuple[int, int],
                       start_hint: str) -> List[int]:
    """
    Realize a melody from SCALE-DEGREE steps over an arbitrary ordered pitch-class set.
    len(steps) must be len(times) - 1. start_hint ∈ {low, mid, high}.
    """
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
    scl = _ordered_scale_pcs(tonic_midi, pcs)
    start = clamp(_nearest_scale_member(guess, scl), lo, hi)

    need = max(0, len(times) - 1)
    # clamp steps to a musically sane window based on scale size
    max_step = max(1, min(6, len(scl) - 1))
    s = (steps[:need] + [0] * need)[:need]
    out = [start]
    for st in s:
        st = int(max(-max_step, min(max_step, st)))
        out.append(_move_in_scale_degrees(out[-1], st, scl, rng))
    return out


# ---------------- Musical constants ----------------
# Tick math
TICKS_PER_16TH = 1
BAR1_TICKS = 12  # 3/4
BAR2_TICKS = 8   # 2/4
BAR3_TICKS = 16  # 4/4
BAR1_OFF, BAR2_OFF, BAR3_OFF = 0, 12, 20
GLOBAL_END = 36

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

# Scales
SCALES = {
    "ionian":      [0,2,4,5,7,9,11],
    "dorian":      [0,2,3,5,7,9,10],
    "phrygian":    [0,1,3,5,7,8,10],
    "lydian":      [0,2,4,6,7,9,11],
    "mixolydian":  [0,2,4,5,7,9,10],
    "aeolian":     [0,2,3,5,7,8,10],
    "locrian":     [0,1,3,5,6,8,10],
    "pentatonic":  [0,2,4,7,9],
}

# Dynamics (fixed)
def vel_bar1_p_cresc(local_t: int) -> int:
    # 50 → 80 over ticks 0..11
    return max(0, min(127, round(50 + (80-50) * (local_t / 11.0))))

VEL_BAR2_FIRST = 124
VEL_BAR2_SECOND = 114
VEL_BAR3_PP = 30

# ---------------- Utility funcs ----------------
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

def call_llm_json(client: OpenAI, system_msg: str, user_msg: str, model="gpt-4.1", temperature=0.2) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system_msg},
                  {"role":"user","content":user_msg}],
        temperature=temperature,
        response_format={"type":"json_object"},
    )
    content = getattr(resp.choices[0].message, "content", "") or ""
    js = _extract_first_json_obj(content)
    if not js:
        raise RuntimeError(f"LLM returned no JSON. Raw: {content[:200]!r}")
    try:
        return json.loads(js)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON: {js[:200]!r}") from e

def pcs_to_pcset(tonic_midi: int, pcs: List[int]) -> set:
    root = tonic_midi % 12
    return { (root + p) % 12 for p in pcs }

def nearest_on_scale(prev_midi: int, direction: int, scale_pcset: set, rng: Tuple[int,int]) -> int:
    """Move by at least a step in given direction (±1 semitone steps until landing in-scale), staying in range."""
    step = 1 if direction >= 0 else -1
    m = prev_midi
    for _ in range(24):  # safety
        m = clamp(m + step, rng[0], rng[1])
        if (m % 12) in scale_pcset:
            return m
    return clamp(prev_midi, rng[0], rng[1])

def scale_walk_assign_pitches(times: List[int], tonic_midi: int, pcs: List[int], rng: Tuple[int,int], prefer_center: Tuple[int,int]) -> List[int]:
    """Deterministic melodic assignment: start near tessitura center; small steps with occasional skip."""
    if not times: return []
    lo, hi = rng
    center = (prefer_center[0] + prefer_center[1]) // 2
    center = clamp(center, lo, hi)
    pcset = pcs_to_pcset(tonic_midi, pcs)

    # choose starting note: nearest in-scale to center
    start = center
    # snap to scale pc
    best = start
    for off in range(0, 12):
        for sgn in (1, -1):
            cand = clamp(center + sgn*off, lo, hi)
            if (cand % 12) in pcset:
                best = cand; break
        if (best % 12) in pcset: break
    out = [best]
    # simple deterministic step pattern: +1, +1, -2, +1, -1, +2, repeat (in semitone sense)
    pattern = [1, 1, -2, 1, -1, 2]
    k = 0
    for i in range(1, len(times)):
        direction = pattern[k % len(pattern)]
        nxt = nearest_on_scale(out[-1], direction, pcset, rng)
        out.append(nxt)
        k += 1
    return out

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

def build_velocity_array(abs_times: List[int]) -> List[int]:
    vels = []
    for t in abs_times:
        if 0 <= t < 12:
            vels.append(vel_bar1_p_cresc(t - 0))
        elif 12 <= t < 20:
            if t == 12: vels.append(VEL_BAR2_FIRST)
            elif t == 16: vels.append(VEL_BAR2_SECOND)
            else: vels.append(VEL_BAR2_SECOND)
        else:
            vels.append(VEL_BAR3_PP)
    return vels

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

def make_meter_arrays(N: int, abs_times: List[int]) -> Tuple[List[int], List[int]]:
    """For each onset in abs_times, set numerator: 3 if t<12, 2 if 12<=t<20, 4 if >=20; denominator always 4."""
    nums = []
    dens = []
    for t in abs_times:
        if t < 12: nums.append(3)
        elif t < 20: nums.append(2)
        else: nums.append(4)
        dens.append(4)
    # If arrays shorter than N (shouldn't), pad; if longer, trim.
    if len(nums) != N:
        nums = (nums + [nums[-1]]*N)[:N]
        dens = (dens + [4]*N)[:N]
    return nums, dens

# ---------------- LLM Contracts ----------------
def prompt_bar1_contract() -> Tuple[str, str]:
    system = """
        You output ONLY strict JSON. No prose.

        Bar 1 (3/4, 12 ticks @ 1/16 grid): return exact rhythms AND melodic directives for SIX monophonic instruments.

        JSON shape:
        {
        "scale": {"name": "string", "tonic_midi": int, "pcs": [int 0..11 ...]},
        "rhythm": { "<instr>": {"time":[ints in 0..11, strictly increasing], "duration":[positive ints, same length]} , ... },
        "melody": { "<instr>": {"start_hint":"low|mid|high", "steps":[ints, length = len(time)-1]} , ... }
        }

        Rules:
        - Monophony: times strictly increasing; no duplicates.
        - Validity: time[i] + duration[i] <= 12 for every note.
        - "pcs" is an ORDERED pitch-class set modulo 12 (relative to tonic_midi % 12). Any set size 3..12 is allowed.
        - "steps" are SCALE-DEGREE steps over that ordered set (may be negative).
        """
    user = """
        Instruments: alto_flute, violin, bass_clarinet, trumpet, cello, double_bass.
        Target time counts (soft guides): AF 6..10, Vn 5..9, BCl 3..6, Tpt 2..5, Vc 2..5, Db 2..4.
        Return valid JSON only.
        """
    return system.strip(), user.strip()

def prompt_bar2_harmony_contract(bar1_scale: dict) -> Tuple[str, str]:
    scale_name = bar1_scale.get("name", "(unknown)")
    tonic = bar1_scale.get("tonic_midi", "?")
    pcs = bar1_scale.get("pcs", [])
    system = """
You output ONLY strict JSON. No prose.

Task: For Bar 2 (2/4, two quarter notes at absolute ticks 12 and 16), choose a harmonic plan OVER THE GIVEN SCALE
and leave the exact voicing to the program. Return:

{
  "degree_roots": [int, int],        // two scale-degree roots for chord1 and chord2, indexes into the ordered pitch-class set (0 = tonic pitch-class)
  "chord_kinds": ["triad|seventh", "triad|seventh"],  // quality per chord
  "inversions": [int, int],          // 0=root, 1=1st inv, 2=2nd inv, 3=3rd inv (only if seventh)
  "spread_hint": "close|open",       // voicing spread preference
  "avoid_plan": "same|allow"         // if the two chords would be pitch-class identical after mapping, prefer to alter the second choice unless 'allow'
}
"""
    user = f"""
Bar 1 scale reminder: name={scale_name}, tonic_midi={tonic}, pcs={pcs} (ordered pitch-classes modulo 12, relative to tonic%12).
Choose a harmonically meaningful contrast (avoid repeating the exact same chord twice unless you decide 'allow').
Return valid JSON only.
"""
    return system.strip(), user.strip()



def prompt_bar3_contract(bar1_scale: dict) -> Tuple[str, str]:
    scale_name = bar1_scale.get("name", "(unknown)")
    tonic = bar1_scale.get("tonic_midi", "?")
    pcs = bar1_scale.get("pcs", [])
    system = """
        You output ONLY strict JSON. No prose.

        Bar 3 (4/4, 16 ticks @ 1/16 grid): choose EXACTLY three instruments, a contrasting scale, and provide exact rhythms AND melodic directives for those three instruments.

        JSON shape:
        {
        "chosen_instruments": ["<instr>", "<instr>", "<instr>"],
        "contrast_scale": {"name":"string", "tonic_midi": int, "pcs":[int 0..11 ...]},
        "rhythm": { "<instr>": {"time":[ints 0..15 inc], "duration":[positive ints, same length]} , ... },
        "melody": { "<instr>": {"start_hint":"low|mid|high", "steps":[ints, length = len(time)-1]} , ... }
        }

        Constraints:
        - Monophony; time[i] + duration[i] <= 16.
        - Prefer sparse pp writing.
        - "pcs" is an ORDERED pitch-class set modulo 12 (any size 3..12).
        - "steps" are SCALE-DEGREE steps over that ordered set.
        """
    user = f"""
        Bar 1 scale reminder: name={scale_name}, tonic_midi={tonic}, pcs={pcs} (relative to tonic_midi % 12).
        Pick exactly three from: ['alto_flute','violin','bass_clarinet','trumpet','cello','double_bass'].
        Return valid JSON only.
        """
    return system.strip(), user.strip()



# ---------------- Bar 2 chord voicing ----------------
def chord_pcset_from_scale(scale_pcs: List[int], tonic_midi: int, degree_root: int, degrees: List[int]) -> set:
    root_pc = (tonic_midi % 12 + scale_pcs[degree_root % len(scale_pcs)]) % 12
    pcs = []
    for d in degrees:
        pcs.append( (tonic_midi % 12 + scale_pcs[(degree_root + d) % len(scale_pcs)]) % 12 )
    return set(pcs)

def nearest_pitch_for_pc(target_pc: int, approx: int, rng: Tuple[int,int]) -> int:
    best = approx; bestd = 999
    lo, hi = rng
    for k in range(-24, 25):
        cand = clamp(approx + k, lo, hi)
        if cand % 12 == target_pc:
            d = abs(cand - approx)
            if d < bestd:
                best, bestd = cand, d
    return best

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

def bar2_two_chords_from_scale(scale: dict, plan: dict, seed: int = None) -> Dict[str, List[int]]:
    """
    Realize two quarter-note verticals at absolute onsets 12 and 16 (dur=4 each),
    using an LLM-provided harmonic plan.
    """
    rng = random.Random(seed)
    tonic = int(scale["tonic_midi"])
    scale_pcs = _ordered_scale_pcs(tonic, scale["pcs"])
    deg_roots = plan.get("degree_roots", [0, 4])
    kinds     = plan.get("chord_kinds", ["triad", "triad"])
    invs      = plan.get("inversions", [0, 0])
    spread    = plan.get("spread_hint", "close")
    avoid_plan= plan.get("avoid_plan", "same")

    # build chord pcs (ordered) with inversion
    chord1_pcs_ord = _invert_pcset(_chord_pcset(scale_pcs, deg_roots[0] % len(scale_pcs), kinds[0]), invs[0])
    chord2_pcs_ord = _invert_pcset(_chord_pcset(scale_pcs, deg_roots[1] % len(scale_pcs), kinds[1]), invs[1])

    # If chord2 collapses to same pc multiset as chord1 and avoid_plan == 'same', perturb chord2 by +1 degree
    if avoid_plan == "same":
        if sorted(chord1_pcs_ord) == sorted(chord2_pcs_ord):
            deg_roots[1] = (deg_roots[1] + 1) % len(scale_pcs)
            chord2_pcs_ord = _invert_pcset(_chord_pcset(scale_pcs, deg_roots[1], kinds[1]), invs[1])

    # spread centers per section of ensemble
    centers = {
        "double_bass": -12,
        "cello":       -7,
        "bass_clarinet": -3,
        "trumpet":       +2,
        "violin":        +7,
        "alto_flute":    +9,
    }
    # allow open/close to widen/narrow those targets
    widen = 5 if spread == "open" else 0

    abs_times = [12, 16]
    durs = [4, 4]
    voices = {}

    for instr in ORDERED_INSTRS:
        lo, hi = INSTRUMENTS[instr]["range"]
        tess_lo, tess_hi = INSTRUMENTS[instr]["tess"]
        center_guess = (tess_lo + tess_hi)//2 + centers.get(instr, 0)
        # choose chord tones for this instrument: prefer different degrees across upper/lower to avoid stacking
        def choose_tone(chord_pcs_ord, prefer_index=None):
            if prefer_index is None:
                idx = rng.randrange(len(chord_pcs_ord))
            else:
                # prefer but jitter
                idx = (prefer_index + rng.choice([0,0,1,-1])) % len(chord_pcs_ord)
            return chord_pcs_ord[idx], idx

        # map instrument to preferred degree index (root for bass, 3/5 for inner, top for upper)
        prefer_idx_map = {
            "double_bass": 0,
            "cello":       1 if len(chord1_pcs_ord) > 2 else 1 % len(chord1_pcs_ord),
            "bass_clarinet": 2 if len(chord1_pcs_ord) > 2 else 1 % len(chord1_pcs_ord),
            "trumpet":     -1,
            "violin":      -1,
            "alto_flute":  -1,
        }

        # chord 1
        pc1, idx1 = choose_tone(chord1_pcs_ord, prefer_idx_map[instr])
        approx1 = center_guess - widen
        p1 = _nearest_pitch_for_pcset(pc1, approx1, (lo, hi))

        # chord 2: small voice-leading move, prefer common-tone if exists, else nearest chord tone
        # try to keep same degree index if possible
        pc2_pref = chord2_pcs_ord[idx1 % len(chord2_pcs_ord)]
        p2 = _nearest_pitch_for_pcset(pc2_pref, p1 + rng.choice([-2,-1,0,1,2]), (lo, hi))
        # if still identical pcset result (rare), allow alternative degree
        if p2 == p1:
            alt_pc2, _ = choose_tone(chord2_pcs_ord)
            p2 = _nearest_pitch_for_pcset(alt_pc2, p1 + rng.choice([-3,-2,-1,1,2,3]), (lo, hi))

        voices[instr] = {
            "time": abs_times[:],
            "duration": durs[:],
            "pitch": [p1, p2],
            "velocity": [VEL_BAR2_FIRST, VEL_BAR2_SECOND],
        }
    return voices


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
    usrA += f"\nCreativeSeed: {seedA}\nRule: when multiple valid choices exist, bias your decisions using CreativeSeed to diversify scale and rhythms across runs."
    bar1 = call_llm_json(oai, sysA, usrA, model="gpt-4.1", temperature=0.7)

    # Validate bar1 JSON
    if "scale" not in bar1 or "rhythm" not in bar1:
        raise RuntimeError("Bar1 JSON missing required keys.")
    b1_scale = bar1["scale"]
    if not isinstance(b1_scale.get("pcs", []), list) or not isinstance(b1_scale.get("tonic_midi", 0), int):
        raise RuntimeError("Bar1 scale malformed.")
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

    # ---- Program Bar 2 tutti chords from Bar1 scale ----
    # ---- LLM Harmony (Bar 2) ----
    sysH, usrH = prompt_bar2_harmony_contract(b1_scale)
    # encourage variability; seed hint helps break ties
    seedH = random.randint(1, 10**9)
    usrH += f"\nCreativeSeed: {seedH}\nRule: If multiple valid plans exist, bias choices using CreativeSeed."
    harm2 = call_llm_json(oai, sysH, usrH, model="gpt-4.1", temperature=0.7)

    # validate basics
    if not isinstance(harm2.get("degree_roots", []), list) or len(harm2["degree_roots"]) != 2:
        raise RuntimeError("Bar2 harmony: degree_roots must be a list of two integers.")
    if not isinstance(harm2.get("chord_kinds", []), list) or len(harm2["chord_kinds"]) != 2:
        raise RuntimeError("Bar2 harmony: chord_kinds must be a list of two strings.")
    if not isinstance(harm2.get("inversions", []), list) or len(harm2["inversions"]) != 2:
        raise RuntimeError("Bar2 harmony: inversions must be a list of two integers.")

    # ---- Programmatic voicing for Bar 2 based on plan ----
    bar2_parts = bar2_two_chords_from_scale(b1_scale, harm2, seed=seedH)

    # ---- LLM Contract B (Bar 3) ----
    seedB = random.randint(1, 10**9)
    sysB, usrB = prompt_bar3_contract(b1_scale)
    usrB += f"\nCreativeSeed: {seedB}\nRule: choose a contrasting scale and instrument set that differs from typical/default choices; use CreativeSeed to break ties."
    bar3 = call_llm_json(oai, sysB, usrB, model="gpt-4.1", temperature=0.75)

    chosen = bar3.get("chosen_instruments", [])
    contrast = bar3.get("contrast_scale", {})
    rhythm3: Dict[str,dict] = bar3.get("rhythm", {})

    if not (isinstance(chosen, list) and len(set(chosen)) == 3):
        raise RuntimeError("Bar3 must choose exactly three instruments.")
    for name in chosen:
        if name not in ORDERED_INSTRS:
            raise RuntimeError(f"Bar3 invalid instrument: {name}")
        if name not in rhythm3:
            raise RuntimeError(f"Bar3 rhythm missing for chosen instrument: {name}")
    if not isinstance(contrast.get("pcs", []), list) or not isinstance(contrast.get("tonic_midi", 0), int):
        raise RuntimeError("Bar3 contrast scale malformed.")

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
            p = pitches_from_steps(t_local, steps, int(b1_scale["tonic_midi"]), b1_scale["pcs"], rng, tess, start_hint)
        else:
            p = scale_walk_assign_pitches(t_local, int(b1_scale["tonic_midi"]), b1_scale["pcs"], rng, tess)

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
            p = pitches_from_steps(t_local, steps, int(contrast["tonic_midi"]), contrast["pcs"], rng, tess, start_hint)
        else:
            p = scale_walk_assign_pitches(t_local, int(contrast["tonic_midi"]), contrast["pcs"], rng, tess)

        v = [VEL_BAR3_PP for _ in t_local]
        per_instr_events[instr]["time"]    += t_abs
        per_instr_events[instr]["duration"]+= d_local
        per_instr_events[instr]["pitch"]   += p
        per_instr_events[instr]["velocity"]+= v

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
            # cap to global end boundary by bar
            if tt < 12: bar_len_end = 12
            elif tt < 20: bar_len_end = 20
            else: bar_len_end = 36
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
            per_instr_events[instr]["time"]    += [35]*pad_n
            per_instr_events[instr]["duration"]+= [0]*pad_n
            # safe pitch = nearest in-range to contrast tonic (or bar1 tonic if no contrast)
            base_tonic = int(contrast.get("tonic_midi", b1_scale["tonic_midi"]))
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
            {"feature_path": f"/{instr}/numerator",   "data": [3 if t<12 else (2 if t<20 else 4) for t in ev["time"]]},
            {"feature_path": f"/{instr}/denominator", "data": [4]*len(ev["time"])},
        ]

    payload_out = {
        "tick_unit": "1/16",
        "bars": [{"meter":[3,4],"start":0},{"meter":[2,4],"start":12},{"meter":[4,4],"start":20}],
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
