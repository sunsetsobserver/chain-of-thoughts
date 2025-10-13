#!/usr/bin/env python3
"""
Chain of Thoughts — One-Bar Composer (Supervised, Multi-Track, Monophonic)
- LLM creates one PT per instrument (role-aware, scale-aware).
- Execute with DCN SDK.
- Supervisor enforces monophony, snaps to scale/range, thins by role,
  and repairs verticals at bar checkpoints (0,8,16,24).
- Emits { tracks: { instrument: [ {feature_path,data}, ... ] } } JSON.
"""

import os, sys, time, json, random, requests
from typing import Dict, List, Tuple
from secrets import OPENAI_API_KEY
from eth_account import Account
from eth_account.messages import encode_defunct
from openai import OpenAI

# --------------------- DCN API helpers (as in Allagma) ---------------------
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

# --------------------- Musical configuration ---------------------
BAR_UNITS = 32
METER_NUM, METER_DEN = 4, 4
CHECKPOINTS = [0, 8, 16, 24]  # strong positions inside the bar

# Sounding MIDI ranges (safe)
INSTRUMENTS = {
    "alto_flute":    {"range": (55, 93),  "tess": (60, 86)},  # G3–A6
    "violin":        {"range": (55, 103), "tess": (60, 96)},  # G3–G7
    "bass_clarinet": {"range": (34, 82),  "tess": (38, 76)},  # D2–A5
    "trumpet":       {"range": (58, 94),  "tess": (62, 88)},  # A#3–A6
    "cello":         {"range": (36, 76),  "tess": (41, 72)},  # C2–E5
    "double_bass":   {"range": (28, 64),  "tess": (31, 57)},  # E1–E4
}

ROLE_BY_INSTR = {
    "alto_flute":    "melody",
    "violin":        "counter",
    "bass_clarinet": "pad",
    "trumpet":       "punct",
    "cello":         "pedal",
    "double_bass":   "pedal",
}

MIN_IOI_BY_ROLE = {"melody":1, "counter":2, "pad":4, "pedal":8, "punct":3}

SCALES = {
    "ionian":      [0,2,4,5,7,9,11],
    "dorian":      [0,2,3,5,7,9,10],
    "aeolian":     [0,2,3,5,7,8,10],
    "mixolydian":  [0,2,4,5,7,9,10],
    "pentatonic":  [0,2,4,7,9],
}

def build_bar_spec(tonic=62, mode="dorian", chord="triad"):
    """
    chord: 'triad' or 'seventh' → relative degrees [0,2,4] or [0,2,4,6]
    degree_roots: modal checkpoints in degrees of the scale (0,3,4,0) works well in Dorian
    """
    degrees = [0,2,4] if chord == "triad" else [0,2,4,6]
    return {
        "tonic": tonic,
        "mode": mode,
        "scale": SCALES[mode],
        "chord_degrees": degrees,
        "degree_roots": [0, 3, 4, 0],  # I - IV - V - I (modal sense)
    }

# --------------------- Small utility functions ---------------------
def clamp(v, lo, hi): return max(lo, min(hi, int(v)))

def nearest_scale_pitch(p: int, tonic: int, scale: List[int]) -> int:
    """Return the nearest pitch to p that lies on the infinite lattice of tonic+scale."""
    best, best_dist = p, 999
    tonic_pc = tonic % 12
    pcs = [(tonic_pc + s) % 12 for s in scale]
    base_oct = p - (p % 12)
    # search a few neighboring octaves
    for oct_shift in (-24, -12, 0, 12, 24, 36, -36):
        for pc in pcs:
            cand = base_oct + pc + oct_shift
            d = abs(cand - p)
            if d < best_dist:
                best, best_dist = cand, d
    return best

def snap_stream_to_scale(pitches: List[int], rng: Tuple[int,int], barspec: dict) -> List[int]:
    lo, hi = rng
    out = []
    for x in pitches:
        snapped = nearest_scale_pitch(int(x), barspec["tonic"], barspec["scale"])
        out.append(clamp(snapped, lo, hi))
    return out

def thin_by_min_ioi(times, pitches, durs, vels, min_ioi=2):
    keep_t, keep_i, last = [], [], None
    for i, t in enumerate(times):
        t = int(t)
        if last is None or (t - last) >= min_ioi:
            keep_t.append(t); keep_i.append(i); last = t
    return (
        keep_t,
        [pitches[i] for i in keep_i],
        [durs[i] for i in keep_i],
        [vels[i] for i in keep_i],
    )

def freeze_meter(dims: List[dict]) -> None:
    by = { (d.get("feature_name") or "").strip().lower(): d for d in dims }
    for meter in ("numerator","denominator"):
        if meter in by:
            by[meter]["transformations"] = [{"name":"add","args":[0]}]
        else:
            dims.append({"feature_name": meter, "transformations":[{"name":"add","args":[0]}]})

import re

def _extract_first_json_obj(text: str) -> str:
    """Return the first top-level {...} JSON object substring from text (or '')."""
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

def _call_llm_json(client, model, system_msg, user_msg, temperature=0.35):
    """
    Call Chat Completions in JSON mode and return a parsed dict.
    Retries once with a fallback model if needed.
    """
    def _one(model_name):
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"system","content":system_msg},
                      {"role":"user","content":user_msg}],
            temperature=temperature,
            response_format={"type": "json_object"},  # <- force JSON
        )
        content = getattr(resp.choices[0].message, "content", "") or ""
        # Some SDKs may return the JSON in tool-calls; we only handle content here
        js = _extract_first_json_obj(content)
        if not js:
            raise ValueError(f"LLM returned no JSON content (model={model_name}). Raw: {content[:200]!r}")
        try:
            return json.loads(js)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from model={model_name}: {js[:200]!r}") from e

    try:
        return _one(model)
    except Exception as e1:
        # Fallback to a widely-available model name
        fallback = "gpt-4o-mini"
        if model != fallback:
            try:
                return _one(fallback)
            except Exception as e2:
                raise RuntimeError(f"Both primary ({model}) and fallback ({fallback}) failed: {e2}") from e1
        raise


# --------------------- LLM feature generator (role + scale aware) ---------------------
def generate_feature_with_llm(client: OpenAI, name_hint: str, instr: str, r_lo:int, r_hi:int,
                              barspec: dict, role: str) -> dict:
    scale_str = ",".join(map(str, barspec["scale"]))
    role_rules = {
        "melody":  "- 5–10 notes; stepwise in SCALE; leaps only to CHORD TONES; one clear arc.\n",
        "counter": "- 3–7 notes; contrary to melody center; chord tones on strong beats.\n",
        "pad":     "- 1–3 long notes; sustained chord tones; sparse rhythm.\n",
        "pedal":   "- 1–2 very long notes; tonic or fifth; almost static.\n",
        "punct":   "- 2–4 short notes; near beats 1/3; chord tones; avoid runs.\n",
    }.get(role, "")

    system_msg = (
        "Output ONLY a single valid JSON object for a DCN PT FEATURE (keys: name, dimensions, feature_name, transformations, args).\n"
        "Allowed ops ONLY: add, subtract (each with exactly one uint arg). Meter frozen: numerator/denominator = add 0.\n"
        "TIME: only add; mostly >0; a FEW add 0 allowed (articulation); total advance 8..32 units; monophonic intent.\n"
        "DURATION: 4–10 ops; 2–3 values via add 0 plateaus; near-zero drift. VELOCITY: 3–8 ops; small arcs; near-zero drift.\n"
        "Strictly valid JSON. No prose.\n"
    )
    user_msg = (
        f"INSTRUMENT: {instr} | SOUNDING MIDI range [{r_lo}..{r_hi}]\n"
        f"BAR SPEC: tonic={barspec['tonic']} | mode={barspec['mode']} | SCALE offsets=[{scale_str}] | chord_degrees={barspec['chord_degrees']} | checkpoints={CHECKPOINTS}\n"
        f"ROLE: {role.upper()}\n{role_rules}"
        "PITCH RULES: Use SCALE ONLY. Mostly stepwise; leaps (4/5/8ve) land on chord tones. Stay within range; near-zero drift.\n"
        "TIME: small positive steps (1–4), occasional longer value; avoid long runs of time add 0 (no chord bursts).\n"
        "OUTPUT SHAPE:\n"
        "{ \"name\":\"...\", \"dimensions\": [\n"
        "  {\"feature_name\":\"pitch\",\"transformations\":[...]},\n"
        "  {\"feature_name\":\"time\",\"transformations\":[...]},\n"
        "  {\"feature_name\":\"duration\",\"transformations\":[...]},\n"
        "  {\"feature_name\":\"velocity\",\"transformations\":[...]},\n"
        "  {\"feature_name\":\"numerator\",\"transformations\":[{\"name\":\"add\",\"args\":[0]}]},\n"
        "  {\"feature_name\":\"denominator\",\"transformations\":[{\"name\":\"add\",\"args\":[0]}]}\n"
        "]}\n"
        f"Name hint: {name_hint}\n"
    )

    # Primary model; adjust if you prefer
    primary_model = "gpt-4.1-mini"

    data = _call_llm_json(
        client=client,
        model=primary_model,
        system_msg=system_msg,
        user_msg=user_msg,
        temperature=0.35,
    )

    # Make name unique & ensure dimensions list
    data["name"] = (data.get("name") or name_hint) + "_" + str(int(time.time()*1000))
    if not isinstance(data.get("dimensions"), list):
        raise ValueError(f"LLM JSON missing 'dimensions' list: {data}")
    return data


# --------------------- Execute PT and get streams ---------------------
def execute_pt_bar(sdk_client, pt_name: str, dims: List[dict], bar_units:int,
                   instr: str, instr_range: Tuple[int,int], barspec: dict, role: str) -> Dict[str, List[int]]:
    seeds = {
        "pitch":  (instr_range[0] + instr_range[1]) // 2,
        "time":   0,
        "duration": 2,
        "velocity": 70,
        "numerator": METER_NUM,
        "denominator": METER_DEN,
    }
    running = [(0,0)] * (1 + len(dims))
    running[0] = (0,0)
    for i, d in enumerate(dims):
        fname = (d.get("feature_name") or "").strip().lower()
        running[i+1] = (int(seeds.get(fname, 0)), 0)

    N = 32
    result = sdk_client.execute(pt_name, N, running)

    streams = {}
    for s in result:
        fp = getattr(s, "feature_path", None) or (s.get("feature_path") if isinstance(s, dict) else "")
        dat = getattr(s, "data", None) or (s.get("data") if isinstance(s, dict) else [])
        tail = (fp.split("/")[-1]).strip().lower() if fp else ""
        streams[tail] = [int(v) for v in dat]

    # defaults
    for k in ["pitch","time","duration","velocity","numerator","denominator"]:
        if k not in streams: streams[k] = []

    # keep only events inside the bar
    times = streams["time"]
    L = min(len(streams["pitch"]), len(times), len(streams["duration"]), len(streams["velocity"]))
    idx_keep = []
    for i in range(L):
        t = int(times[i])
        if 0 <= t <= (bar_units - 1): idx_keep.append(i)
        elif t > (bar_units - 1): break

    if not idx_keep:
        return {k:[] for k in ["pitch","time","duration","velocity","numerator","denominator"]}

    pp = [streams["pitch"][i] for i in idx_keep]
    tt = [int(streams["time"][i]) for i in idx_keep]
    dd = [max(0, int(streams["duration"][i])) for i in idx_keep]
    vv = [max(0, min(127, int(streams["velocity"][i]))) for i in idx_keep]

    # snap to scale & range
    pp = snap_stream_to_scale(pp, instr_range, barspec)
    # thin by role
    min_ioi = MIN_IOI_BY_ROLE.get(role, 2)
    tt, pp, dd, vv = thin_by_min_ioi(tt, pp, dd, vv, min_ioi=min_ioi)

    return {
        "pitch": pp,
        "time": tt,
        "duration": dd,
        "velocity": vv,
        "numerator": [METER_NUM]*len(tt),
        "denominator": [METER_DEN]*len(tt),
    }

# --------------------- Data shapers for supervisor ---------------------
def to_notes(streams: Dict[str, List[int]]) -> List[dict]:
    """Convert scalar arrays to a list of monophonic notes [{t,p,d,v}] sorted by t."""
    notes = []
    L = min(len(streams["time"]), len(streams["pitch"]), len(streams["duration"]), len(streams["velocity"]))
    for i in range(L):
        notes.append({"t": int(streams["time"][i]),
                      "p": int(streams["pitch"][i]),
                      "d": max(1, int(streams["duration"][i])),
                      "v": max(1, min(127, int(streams["velocity"][i])))})
    return sorted(notes, key=lambda e: e["t"])

def from_notes(notes: List[dict]) -> Dict[str, List[int]]:
    return {
        "time": [n["t"] for n in notes],
        "pitch": [n["p"] for n in notes],
        "duration": [n["d"] for n in notes],
        "velocity": [n["v"] for n in notes],
        "numerator": [METER_NUM]*len(notes),
        "denominator": [METER_DEN]*len(notes),
    }

# --------------------- Supervisor: monophony + harmony repairs ---------------------
def enforce_strict_monophony(notes: List[dict], min_ioi: int) -> List[dict]:
    """Ensure strictly increasing onsets and ≥ min_ioi between notes for a single instrument."""
    out, last_t = [], None
    for n in sorted(notes, key=lambda x: x["t"]):
        t = n["t"]
        if last_t is None or t >= last_t + min_ioi:
            out.append(n); last_t = t
        else:
            # Try to nudge right
            new_t = last_t + min_ioi
            if new_t <= BAR_UNITS - 1:
                out.append({**n, "t": new_t})
                last_t = new_t
            # else drop
    return out

def chord_pc_set(tonic_pc: int, scale: List[int], root_degree: int, chord_degrees: List[int]) -> set:
    degs = [ (root_degree + k) % len(scale) for k in chord_degrees ]
    return { (tonic_pc + scale[d]) % 12 for d in degs }

def nearest_pc_pitch(target_pc: int, guess_p: int) -> int:
    """Move guess to nearest pitch class target_pc (± a few semitones)."""
    candidates = [guess_p + k for k in (-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12)]
    best, bestd = guess_p, 999
    for c in candidates:
        if c % 12 == target_pc:
            d = abs(c - guess_p)
            if d < bestd:
                best, bestd = c, d
    return best

def pick_root_midi(tonic: int, scale: List[int], root_degree: int, approx: int, rng: Tuple[int,int]) -> int:
    """Return the chord root (on given degree) nearest to approx within range."""
    root_pc = (tonic % 12 + scale[root_degree]) % 12
    cand = nearest_pc_pitch(root_pc, approx)
    return clamp(cand, rng[0], rng[1])

def get_note_at_time(notes: List[dict], t: int, tol: int = 0) -> Tuple[int, dict]:
    """Return (index, note) where note starts at t (within ±tol). If multiple, pick nearest; else (-1, None)."""
    best_i, best, best_d = -1, None, 999
    for i, n in enumerate(notes):
        d = abs(n["t"] - t)
        if d <= tol and d < best_d:
            best_i, best, best_d = i, n, d
    return best_i, best

def micro_offset_cluster(notes_by_instr: Dict[str, List[dict]], t: int):
    """If ≥4 instruments start exactly at t, shift one mid-voice by +1 (rotate preference)."""
    crowded = [k for k,ns in notes_by_instr.items() if any(n["t"] == t for n in ns)]
    if len(crowded) >= 4:
        for cand in ["bass_clarinet","trumpet","violin","alto_flute"]:
            if cand in crowded:
                ns = notes_by_instr[cand]
                for n in ns:
                    if n["t"] == t and n["t"] + 1 < BAR_UNITS:
                        n["t"] += 1
                        return

def vertical_minor_second_avoid(notes_at_t: List[Tuple[str,dict]], barspec: dict):
    """Avoid m2 at exact checkpoint: nudge one upper voice by +1 time or to nearest chord tone."""
    if len(notes_at_t) < 2: return
    # Sort by pitch low→high
    notes_at_t.sort(key=lambda kv: kv[1]["p"])
    for i in range(len(notes_at_t)-1):
        a_name, a = notes_at_t[i]
        b_name, b = notes_at_t[i+1]
        if abs(a["p"] - b["p"]) == 1:
            # Nudge the upper voice (prefer non-flute)
            target = b
            if b_name == "alto_flute" and a_name != "alto_flute":
                target = a
            if target["t"] + 1 < BAR_UNITS:
                target["t"] += 1
            else:
                # Try pitch nudge to nearest scale tone (already on scale) -> nudge by diatonic step
                step = 2
                target["p"] = nearest_scale_pitch(target["p"] + step, barspec["tonic"], barspec["scale"])

def checkpoint_supervision(notes_by_instr: Dict[str, List[dict]], barspec: dict):
    """At t in CHECKPOINTS: ensure root present (DB/Cello), guide tone presence, avoid m2 stacks, micro-offset."""
    tonic_pc = barspec["tonic"] % 12
    scale = barspec["scale"]
    degrees = barspec["degree_roots"]
    chord_degrees = barspec["chord_degrees"]

    for j, t in enumerate(CHECKPOINTS):
        root_degree = degrees[j % len(degrees)]
        pcs = chord_pc_set(tonic_pc, scale, root_degree, chord_degrees)

        # 1) Root on DB (prefer) / Cello
        for low in ["double_bass", "cello"]:
            ns = notes_by_instr.get(low, [])
            i, n = get_note_at_time(ns, t, tol=2)
            if n:
                rng = INSTRUMENTS[low]["range"]
                approx = n["p"]
                root_p = pick_root_midi(barspec["tonic"], scale, root_degree, approx, rng)
                ns[i]["p"] = root_p
                ns[i]["t"] = t  # align on the checkpoint
                break

        # 2) Ensure at least one guide tone (3rd for triad; 3rd or 7th for seventh)
        want_guide_indices = [2] if chord_degrees == [0,2,4] else [2,6]
        guide_pcs = { (tonic_pc + scale[(root_degree + gi) % len(scale)]) % 12 for gi in want_guide_indices }

        # find any upper instrument active at t
        upper_names = ["bass_clarinet","trumpet","violin","alto_flute"]
        active = []
        for name in upper_names:
            ns = notes_by_instr.get(name, [])
            i, n = get_note_at_time(ns, t, tol=1)
            if n: active.append((name,i,n))

        has_guide = any( (n["p"] % 12) in guide_pcs for _,_,n in active )
        if not has_guide and active:
            # nudge the closest pitch to a guide tone
            name,i,n = min(active, key=lambda tup: min(abs((tup[2]["p"]%12) - gp) for gp in guide_pcs))
            # pick nearest guide pc
            gp = min(guide_pcs, key=lambda pc: min(abs((n["p"]%12) - pc), 12-abs((n["p"]%12) - pc)))
            notes_by_instr[name][i]["p"] = nearest_pc_pitch(gp, n["p"])

        # 3) Avoid vertical minor seconds at t
        vertical = []
        for name in INSTRUMENTS.keys():
            ns = notes_by_instr.get(name, [])
            _, n = get_note_at_time(ns, t, tol=0)
            if n: vertical.append((name, n))
        vertical_minor_second_avoid(vertical, barspec)

        # 4) Micro-offset if crowded
        micro_offset_cluster(notes_by_instr, t)

# --------------------- Main composition function ---------------------
def compose_one_bar(context_prompt: str = "", tonic: int = 62, mode: str = "dorian", chord: str = "triad") -> dict:
    # --- Auth ---
    acct = _get_account()
    nonce = get_nonce(API_BASE, acct.address)
    msg = f"Login nonce: {nonce}"
    sig = acct.sign_message(encode_defunct(text=msg)).signature.hex()
    auth = post_auth(API_BASE, acct.address, msg, sig)
    access = auth.get("access_token"); refresh = auth.get("refresh_token")
    if not access or not refresh: raise RuntimeError("Auth tokens missing")
    _ = post_refresh(API_BASE, access, refresh)

    # --- LLM client ---
    oai = OpenAI(api_key=OPENAI_API_KEY)

    # --- Bar tonal frame ---
    barspec = build_bar_spec(tonic=tonic, mode=mode, chord=chord)

    # --- Per-instrument PT generation + registry upload ---
    vocab_pts = {}
    for instr, prof in INSTRUMENTS.items():
        lo, hi = prof["range"]
        role   = ROLE_BY_INSTR[instr]
        name_hint = f"cot_{instr}_bar"
        feature_json = generate_feature_with_llm(oai, name_hint, instr, lo, hi, barspec, role)
        dims = feature_json.get("dimensions", [])
        if not isinstance(dims, list): dims = []
        freeze_meter(dims)
        feature_json["dimensions"] = dims
        post_feature(API_BASE, access, refresh, feature_json)
        vocab_pts[instr] = {"name": feature_json["name"], "dims": dims, "range": (lo,hi), "role": role}

    # --- Execute each PT ---
    try:
        import dcn
    except Exception as e:
        raise RuntimeError("dcn SDK not available. Please install with: pip install dcn") from e

    sdk = dcn.Client()
    sdk.login_with_account(acct)

    raw_streams = {}
    for instr, meta in vocab_pts.items():
        lo, hi = meta["range"]
        role   = meta["role"]
        raw_streams[instr] = execute_pt_bar(
            sdk, meta["name"], meta["dims"], BAR_UNITS,
            instr=instr, instr_range=(lo,hi), barspec=barspec, role=role
        )

    # --- Supervisor: per-instrument monophony + harmony repairs ---
    # Convert to notes for editing
    notes_by_instr = {k: to_notes(v) for k,v in raw_streams.items()}

    # Ensure strict monophony (≥ min IOI)
    for instr, notes in notes_by_instr.items():
        role = ROLE_BY_INSTR[instr]
        notes_by_instr[instr] = enforce_strict_monophony(notes, MIN_IOI_BY_ROLE.get(role, 2))

    # Checkpoint voicing / micro-offset / vertical cleanup
    checkpoint_supervision(notes_by_instr, barspec)

    # Re-snap to scale & clamp to range (in case repairs moved pitches)
    for instr, notes in notes_by_instr.items():
        rng = INSTRUMENTS[instr]["range"]
        for n in notes:
            n["p"] = clamp(nearest_scale_pitch(n["p"], barspec["tonic"], barspec["scale"]), rng[0], rng[1])

    # Convert back to streams
    final_streams = {instr: from_notes(notes) for instr, notes in notes_by_instr.items()}

    # --- Build output payload compatible with your visualiser ---
    tracks = {}
    for instr, streams in final_streams.items():
        tracks[instr] = [
            {"feature_path": f"/{instr}/pitch",       "data": streams["pitch"]},
            {"feature_path": f"/{instr}/time",        "data": streams["time"]},
            {"feature_path": f"/{instr}/duration",    "data": streams["duration"]},
            {"feature_path": f"/{instr}/velocity",    "data": streams["velocity"]},
            {"feature_path": f"/{instr}/numerator",   "data": streams["numerator"]},
            {"feature_path": f"/{instr}/denominator", "data": streams["denominator"]},
        ]

    payload = {
        "bar_units": BAR_UNITS,
        "meter": [METER_NUM, METER_DEN],
        "context_prompt": context_prompt,
        "barspec": barspec,
        "pt_names": {k:v["name"] for k,v in vocab_pts.items()},
        "tracks": tracks
    }

    with open("cot_bar_payload.json","w",encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("Wrote one-bar payload → cot_bar_payload.json")

    return payload

# --------------------- CLI ---------------------
if __name__ == "__main__":
    # Usage:
    #   python cot_one_bar_supervised.py
    # Optional args: tonic (MIDI), mode, chord
    #   python cot_one_bar_supervised.py 62 dorian triad
    tonic = int(sys.argv[1]) if len(sys.argv) > 1 else 62
    mode  = sys.argv[2] if len(sys.argv) > 2 else "dorian"
    chord = sys.argv[3] if len(sys.argv) > 3 else "triad"
    compose_one_bar("", tonic=tonic, mode=mode, chord=chord)
