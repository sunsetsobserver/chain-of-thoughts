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

def clamp(v, lo, hi): return max(lo, min(hi, int(v)))

def generate_feature_with_llm(client: OpenAI, name_hint: str, instr: str, r_lo:int, r_hi:int) -> dict:
    """
    Ask the LLM for a single PT JSON (monophonic line) tailored to instrument range,
    1 bar long, with meter frozen to 4/4, time advancing inside BAR_UNITS.
    """
    system_msg = (
        "ROLE:\n"
        "Output ONLY a single valid JSON object describing a DCN PT FEATURE DEFINITION. "
        "No prose, no comments, no code fences.\n\n"
        "SHAPE:\n"
        "{\n"
        '  "name": "<string>",\n'
        '  "dimensions": [\n'
        '    {"feature_name":"pitch","transformations":[{"name":"add","args":[<uint32>]} or {"name":"subtract","args":[<uint32>]} ...]},\n'
        '    {"feature_name":"time","transformations":[...only add... ]},\n'
        '    {"feature_name":"duration","transformations":[...]},\n'
        '    {"feature_name":"velocity","transformations":[...]},\n'
        '    {"feature_name":"numerator","transformations":[{"name":"add","args":[0]}]},\n'
        '    {"feature_name":"denominator","transformations":[{"name":"add","args":[0]}]}\n'
        "  ]\n"
        "}\n\n"
        "HARD CONSTRAINTS:\n"
        "- Allowed transformation names ONLY: add, subtract. Each has exactly one unsigned integer arg (0–127). "
        "- Freeze meter: numerator=add 0, denominator=add 0 (exactly one op each).\n"
        "- time: ONLY 'add', majority >0, may include a SMALL number of 'add 0' for articulation (but keep monophonic; no chord stacking). "
        f"- Ensure the TOTAL time advance for one pass of the time cycle stays within 1..{BAR_UNITS}.\n"
        "- Monophonic texture (MELODY ONLY) – no deliberate chord bursts (avoid long runs of time add 0).\n"
        "- pitch: design a near-zero-drift cycle; guarantee that sampling ≤64 steps from ANY seed in a safe mid-range "
        f"keeps values within MIDI [{r_lo}..{r_hi}]. Use corrective subtract/add to bound excursions.\n"
        "- duration: 4–10 ops, few distinct states (2–3) via small toggles and many add 0 plateaus; near-zero net drift.\n"
        "- velocity: 3–8 ops, small arcs or terraces with small steps (0–8), near-zero net drift.\n"
        "- Keep JSON compact and strictly valid. Keys allowed ONLY: name, dimensions, feature_name, transformations, args.\n"
    )

    user_msg = (
        "TASK: Generate ONE bar feature (monophonic) for the instrument given below.\n"
        f"Instrument: {instr}\n"
        "Meter: 4/4 (numerator, denominator both add 0). Time grid is abstract units; one bar target length ≤ 32 units.\n"
        "Keep time strictly non-decreasing with small positive steps; allow a FEW add 0 as articulation but avoid long runs.\n"
        "Pitch must stay inside the instrument's SOUNDING MIDI range and avoid drift (use corrective moves).\n"
        "Return ONLY the JSON object as specified.\n"
        f"Name hint: {name_hint}\n"
    )

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"system","content":system_msg},
                  {"role":"user","content":user_msg}],
        temperature=0.6,
    )
    raw = resp.choices[0].message.content.strip()
    data = json.loads(raw)
    # Uniquify name
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

    vocab_pts = {}   # instr -> {'name':..., 'dims':[...]}

    for instr, prof in INSTRUMENTS.items():
        lo, hi = prof["range"]
        name_hint = f"cot_{instr}_bar"
        feature_json = generate_feature_with_llm(oai, name_hint, instr, lo, hi)
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
