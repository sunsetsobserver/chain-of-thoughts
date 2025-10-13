#!/usr/bin/env python3
import os, sys, time, json, random, requests
from typing import Dict, List, Tuple
from secrets import OPENAI_API_KEY
from eth_account import Account
from eth_account.messages import encode_defunct
from openai import OpenAI

# === Reuse these from your Allagma script (paste them unchanged or import) ===
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
# === end reuse ===

# ---- Chain of Thoughts: one-bar generator (multi-track) ----

BAR_UNITS = 32              # grid length of one bar (time units, arbitrary)
METER_NUM, METER_DEN = 4, 4 # fixed for now

# Sounding MIDI ranges (approx, conservative), monophonic lines:
INSTRUMENTS = {
    "alto_flute":    {"range": (55, 93),  "tess": (60, 86)},  # ~G3–A6; prefer C4–A6
    "violin":        {"range": (55, 103), "tess": (60, 96)},  # G3–G7; prefer C4–C7-
    "bass_clarinet": {"range": (34, 82),  "tess": (38, 76)},  # sounding D2–A5 (Bb BC ~ + M2 + 8 up when written)
    "trumpet":       {"range": (58, 94),  "tess": (62, 88)},  # sounding A#3/Bb3–A6; prefer D4–F6
    "cello":         {"range": (36, 76),  "tess": (41, 72)},  # C2–E5; prefer E2–C5
    "double_bass":   {"range": (28, 64),  "tess": (31, 57)},  # E1–E4; prefer G1–A3
}

ALLOWED_FEATURES = ["pitch","time","duration","velocity","numerator","denominator"]

# --- Add near the top (after INSTRUMENTS etc.) ---

import math

SCALES = {
    "ionian":      [0,2,4,5,7,9,11],
    "dorian":      [0,2,3,5,7,9,10],
    "aeolian":     [0,2,3,5,7,8,10],
    "mixolydian":  [0,2,4,5,7,9,10],
    "pentatonic":  [0,2,4,7,9],
}

DEFAULT_MODE = "dorian"   # easy “interesting but consonant”
DEFAULT_TONIC = 62        # D4
CHORD_TONES = {           # triads/7ths as degree indices into the scale above
    "triad":   [0,2,4],
    "seventh": [0,2,4,6],
}

ROLE_BY_INSTR = {
    "alto_flute":    "melody",
    "violin":        "counter",
    "bass_clarinet": "pad",
    "trumpet":       "punct",
    "cello":         "pedal",
    "double_bass":   "pedal",
}

def build_bar_spec(tonic=DEFAULT_TONIC, mode=DEFAULT_MODE, chord="triad"):
    """Defines the tonal frame for this bar."""
    scale = SCALES[mode]
    chord_degrees = CHORD_TONES[chord]
    return {
        "tonic": tonic,
        "mode": mode,
        "scale": scale,                 # in semitones from tonic
        "chord_degrees": chord_degrees, # indices into scale
    }

def nearest_scale_pitch(p, tonic, scale):
    """Snap MIDI pitch p to nearest pitch in the (infinite) stack of 'tonic + scale'."""
    # Build a reasonable window around p:
    best = p
    best_dist = 128
    # Cover ±2 octaves around p to be safe
    for k in range(-24, 25):
        cand = tonic + k
        # shift cand into scale by replacing its pitch-class with nearest scale member
        root_pc = (tonic % 12)
        scale_pcs = [ (root_pc + s) % 12 for s in scale ]
        # pick the pitch class in this octave closest to cand
        # compute same-octave candidates:
        pc_cand = cand % 12
        # direct match?
        if pc_cand in scale_pcs:
            dist = abs(cand - p)
            if dist < best_dist:
                best, best_dist = cand, dist
        else:
            # try replacing pc with each scale pc at same octave
            base_oct = cand - pc_cand
            for spc in scale_pcs:
                q = base_oct + spc
                dist = abs(q - p)
                if dist < best_dist:
                    best, best_dist = q, dist
    return int(best)

def snap_stream_to_scale(pitches, instr_range, barspec):
    lo, hi = instr_range
    out = []
    for x in pitches:
        snapped = nearest_scale_pitch(int(x), barspec["tonic"], barspec["scale"])
        out.append(max(lo, min(hi, snapped)))
    return out


def clamp(v, lo, hi): return max(lo, min(hi, int(v)))

def generate_feature_with_llm(client: OpenAI, name_hint: str, instr: str, r_lo:int, r_hi:int,
                              barspec: dict, role: str) -> dict:
    """
    Role- and scale-constrained PT JSON for one monophonic bar.
    """
    scale_str = ",".join(map(str, barspec["scale"]))  # e.g., "0,2,3,5,7,9,10"
    chord_deg = barspec["chord_degrees"]              # e.g., [0,2,4]
    role_rules = {
        "melody":  "- Density: 5–10 notes; mostly stepwise within the SCALE; leaps allowed ONLY to CHORD TONES.\n"
                   "- Contour: one clear arc (rise then fall or vice versa). Avoid zig-zag.\n",
        "counter": "- Density: 3–7 notes; move contrary to melody (assume melody center near tonic+7).\n"
                   "- Prefer chord tones on strong beats; steps on weak beats.\n",
        "pad":     "- Density: 1–3 longer notes; prefer sustained chord tones. Avoid busy rhythm.\n",
        "pedal":   "- Density: 1–2 notes; aim near tonic or fifth; long durations; minimal movement.\n",
        "punct":   "- Density: 2–4 short notes; align near beat 1 or 3; choose chord tones; avoid runs.\n"
    }.get(role, "")

    system_msg = (
        "ROLE:\n"
        "Output ONLY a single valid JSON object describing a DCN PT FEATURE DEFINITION (composite), "
        "with keys: name, dimensions, feature_name, transformations, args. No prose.\n\n"
        "HARD CONSTRAINTS:\n"
        "- Transformations allowed ONLY: add, subtract; exactly one unsigned integer arg (0–127).\n"
        "- Meter frozen: numerator:[{\"name\":\"add\",\"args\":[0]}], denominator:[{\"name\":\"add\",\"args\":[0]}].\n"
        "- TIME: only 'add'; mostly >0; a FEW 'add 0' permitted for articulation (monophony, no chord bursts).\n"
        "- TOTAL time advance per cycle: 8..32 units.\n"
        "- DURATION: 4–10 ops; 2–3 distinct values; plateaus via add 0; near-zero drift.\n"
        "- VELOCITY: 3–8 ops; small arcs/terraces (0–8 deltas); near-zero drift.\n"
        "- JSON must be compact and strictly valid.\n"
    )

    user_msg = (
        f"INSTRUMENT (sounding MIDI): {instr} | range [{r_lo}..{r_hi}]\n"
        f"BAR SPEC:\n"
        f"- Tonic MIDI: {barspec['tonic']}\n"
        f"- Mode: {barspec['mode']} with SCALE offsets from tonic: [{scale_str}] (use ONLY these pitch classes).\n"
        f"- Chord degrees (indices into SCALE): {barspec['chord_degrees']} → chord tones.\n"
        f"- ROLE: {role.upper()}\n"
        f"{role_rules}"
        "PITCH RULES:\n"
        "- Use SCALE ONLY (diatonic to the given mode/tonic). Do NOT introduce out-of-scale pitch-classes.\n"
        "- Steps should mostly be diatonic seconds; leaps (4th/5th/oct) should LAND on chord tones.\n"
        "- Keep near-zero drift and keep all values firmly within the instrument range.\n"
        "- Avoid more than 3 consecutive small steps in the SAME direction; insert corrective step or hold.\n"
        "- Prefer notes around the instrument's comfortable tessitura; do not hover at extremes.\n\n"
        "TIME & TEXTURE:\n"
        "- Monophonic; avoid any pattern that implies vertical stacks.\n"
        "- IOIs: prefer small positive steps (1–4 units), with occasional longer value for cadential feel.\n\n"
        "OUTPUT SHAPE (example keys only; no comments in output):\n"
        "{\n"
        '  \"name\": \"...\",\n'
        '  \"dimensions\": [\n'
        '    {\"feature_name\":\"pitch\",\"transformations\":[...]},\n'
        '    {\"feature_name\":\"time\",\"transformations\":[...]},\n'
        '    {\"feature_name\":\"duration\",\"transformations\":[...]},\n'
        '    {\"feature_name\":\"velocity\",\"transformations\":[...]},\n'
        '    {\"feature_name\":\"numerator\",\"transformations\":[{\"name\":\"add\",\"args\":[0]}]},\n'
        '    {\"feature_name\":\"denominator\",\"transformations\":[{\"name\":\"add\",\"args\":[0]}]}\n'
        "  ]\n"
        "}\n"
        f"Name hint: {name_hint}\n"
    )

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"system","content":system_msg},
                  {"role":"user","content":user_msg}],
        temperature=0.4,
    )
    raw = resp.choices[0].message.content.strip()
    data = json.loads(raw)
    data["name"] = (data.get("name") or f"{name_hint}") + "_" + str(int(time.time()*1000))
    return data


def freeze_meter(dims: List[dict]) -> None:
    by = { (d.get("feature_name") or "").strip().lower(): d for d in dims }
    for meter in ("numerator","denominator"):
        if meter in by:
            by[meter]["transformations"] = [{"name":"add","args":[0]}]
        else:
            dims.append({"feature_name": meter, "transformations":[{"name":"add","args":[0]}]})

def execute_pt_bar(sdk_client, pt_name: str, dims: List[dict], bar_units:int) -> Dict[str, List[int]]:
    """
    Execute a PT, then trim/clamp to one bar and normalize streams.
    """
    # Seeds chosen to keep things stable; time starts at 0
    seeds = {
        "pitch": random.randint(50, 70),
        "time": 0,
        "duration": random.choice([1,2,4,8]),
        "velocity": random.randint(50, 80),
        "numerator": METER_NUM,
        "denominator": METER_DEN,
    }
    # RunningInstances: root + one per dimension
    running = [(0,0)] * (1 + len(dims))
    running[0] = (0,0)
    for i, d in enumerate(dims):
        fname = (d.get("feature_name") or "").strip().lower()
        running[i+1] = (int(seeds.get(fname, 0)), 0)

    # Choose N long enough to fill the bar without exploding
    N = 32
    result = sdk_client.execute(pt_name, N, running)

    # normalize to dict arrays
    streams = {}
    for s in result:
        fp = getattr(s, "feature_path", None) or (s.get("feature_path") if isinstance(s, dict) else "")
        dat = getattr(s, "data", None) or (s.get("data") if isinstance(s, dict) else [])
        tail = (fp.split("/")[-1]).strip().lower() if fp else ""
        streams[tail] = [int(v) for v in dat]

    # Ensure required streams; if missing, fill safe defaults
    for k in ["pitch","time","duration","velocity","numerator","denominator"]:
        if k not in streams: streams[k] = []

    # Trim to the bar: clip all events whose time > bar_units-1
    times = streams["time"]
    if not times:
        return {k:[] for k in ["pitch","time","duration","velocity","numerator","denominator"]}
    L = min(len(streams["pitch"]), len(times), len(streams["duration"]), len(streams["velocity"]))
    idx_keep = []
    for i in range(L):
        t = int(times[i])
        if t < 0: continue
        if t <= (bar_units - 1): idx_keep.append(i)
        else: break

    def take(arr): return [int(arr[i]) for i in idx_keep] if arr else []
    
    out = {
        "pitch": take(streams["pitch"]),
        "time":  take(streams["time"]),
        "duration": [max(0, int(v)) for v in take(streams["duration"])],
        "velocity": [max(0, min(127, int(v))) for v in take(streams["velocity"])],
        "numerator": [METER_NUM]*len(idx_keep),
        "denominator": [METER_DEN]*len(idx_keep),
    }
    return out

def compose_one_bar(context_prompt: str = "") -> dict:
    """
    Compose one bar for all instruments; returns a JSON with per-track streams.
    """
    # 1) Auth (same account as Allagma)
    acct = _get_account()
    nonce = get_nonce(API_BASE, acct.address)
    msg = f"Login nonce: {nonce}"
    sig = acct.sign_message(encode_defunct(text=msg)).signature.hex()
    auth = post_auth(API_BASE, acct.address, msg, sig)
    access = auth.get("access_token"); refresh = auth.get("refresh_token")
    if not access or not refresh: raise RuntimeError("Auth tokens missing")
    # Optional refresh
    _ = post_refresh(API_BASE, access, refresh)

    # 2) LLM + /feature per instrument
    oai = OpenAI(api_key=OPENAI_API_KEY)

    barspec = build_bar_spec(tonic=62, mode="dorian", chord="triad")

    vocab_pts = {}   # instr -> {'name':..., 'dims':[...]}

    for instr, prof in INSTRUMENTS.items():
        lo, hi = prof["range"]
        role   = ROLE_BY_INSTR[instr]
        name_hint = f"cot_{instr}_bar"

        feature_json = generate_feature_with_llm(oai, name_hint, instr, lo, hi, barspec, role)
        # Enforce meter freeze/sanity
        try:
            dims = feature_json.get("dimensions", [])
            if not isinstance(dims, list): dims = []
            freeze_meter(dims)
            feature_json["dimensions"] = dims
        except Exception:
            pass

        # Post to DCN
        post_feature(API_BASE, access, refresh, feature_json)
        vocab_pts[instr] = {"name": feature_json["name"], "dims": feature_json["dimensions"]}

    # 3) Execute each PT and build per-track streams
    import dcn
    sdk = dcn.Client()
    sdk.login_with_account(acct)

    tracks = {}
    for instr, meta in vocab_pts.items():
        streams = execute_pt_bar(sdk, meta["name"], meta["dims"], BAR_UNITS)
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
        "pt_names": {k:v["name"] for k,v in vocab_pts.items()},
        "tracks": tracks
    }

    with open("cot_bar_payload.json","w",encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("Wrote one-bar payload → cot_bar_payload.json")

    return payload

if __name__ == "__main__":
    # Optional textual context to steer the LLM (e.g., “contrast staccato from previous bar…”)
    ctx = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
    compose_one_bar(ctx)
