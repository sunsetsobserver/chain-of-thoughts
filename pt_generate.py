"""
pt_generate.py — generate ONE UNIT (one prompt file) using GPT-5 + DCN.
The prompt may yield a SINGLE bar (legacy) or a SECTION with many bars:
- If the model returns {"features":..., "run_plan":...}: treat as one bar.
- If it returns {"bars":[ {...}, {...}, ... ]}: treat as multi-bar section.

We register+execute each bar separately, measure its length, then concatenate
bars back-to-back inside the unit and return a combined per-instrument payload.
"""

import os, json, time, pathlib, importlib.util
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from eth_account import Account
from eth_account.messages import encode_defunct
from openai import OpenAI

from pt_config import (
    API_BASE, ORDERED_INSTRS, INSTRUMENTS, INSTRUMENT_META,
    BAR_TICKS_BY_BAR, meter_from_ticks
)
from pt_prompts import load_system_prompt, render_user_prompt_from_file, PROMPTS_DIR
from dcn_client import DCNClient

# ---------- local helpers ----------
def _runs_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "runs"

def _make_run_dir(label: str) -> pathlib.Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = _runs_root() / f"{ts}_{label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def _save_json(path: pathlib.Path, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _save_text(path: pathlib.Path, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def _normalize_exec(samples_list: List[dict]) -> Dict[str, List[int]]:
    st: Dict[str, List[int]] = {}
    for s in samples_list:
        tail = (s["feature_path"].split("/")[-1] if s["feature_path"] else "").strip().lower()
        st.setdefault(tail, [])
        st[tail] = list(s["data"])
    return st

def _instrument_pack(instrument: str, streams: Dict[str, List[int]]) -> List[dict]:
    packed: List[dict] = []
    for scalar in ("pitch","time","duration","velocity","numerator","denominator"):
        if scalar in streams:
            packed.append({"feature_path": f"/{instrument}/{scalar}", "data": list(streams[scalar])})
    return packed

def _rand_hex(n: int = 4) -> str:
    import os as _os
    return _os.urandom(n).hex()

def _make_unique_name(instr: str, bar: int, role: str, seen: set) -> str:
    def slug(x: str) -> str:
        return ''.join(ch if ch.isalnum() else '_' for ch in (x or ''))
    base = f"{slug(instr)}_{slug(role)}_b{int(bar)}"
    while True:
        nonce = f"{int(time.time()*1000)}_{_rand_hex(2)}"
        name = f"{base}_{nonce}"
        if name not in seen:
            seen.add(name)
            return name

def _load_openai_key() -> str:
    here = pathlib.Path(__file__).resolve().parent
    secrets_path = here / "secrets.py"
    if secrets_path.exists():
        spec = importlib.util.spec_from_file_location("secrets", secrets_path)
        secrets_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(secrets_mod)
        key = getattr(secrets_mod, "OPENAI_API_KEY", None)
        if not key:
            raise RuntimeError("secrets.py present but OPENAI_API_KEY missing.")
        print(f"Loaded OPENAI API key from {secrets_path}")
        return key
    key_env = os.getenv("OPENAI_API_KEY")
    if not key_env:
        raise RuntimeError("No OPENAI_API_KEY found (secrets.py or env).")
    print("Loaded OPENAI_API_KEY from environment.")
    return key_env

# ---------- validation ----------
def _validate_bundle(bundle: Dict[str, Any], bar_label: str):
    ALLOWED = {"add", "subtract", "mul", "div"}
    for feat in bundle.get("features", []):
        pt = feat.get("pt", {})
        for dim in pt.get("dimensions", []):
            fname = (dim.get("feature_name") or "").strip().lower()
            for tr in dim.get("transformations", []):
                op = (tr.get("name") or "").strip().lower()
                if op not in ALLOWED:
                    raise RuntimeError(f"[{bar_label}] Forbidden op '{op}' in feature '{pt.get('name')}', dimension '{fname}'.")
            if fname == "time":
                for tr in dim.get("transformations", []):
                    if (tr.get("name") or "").lower() != "add":
                        raise RuntimeError(f"[{bar_label}] time must use only 'add'. Offender in '{pt.get('name')}'.")
                    a = int(tr.get("args", [0])[0])
                    if a <= 0:
                        raise RuntimeError(f"[{bar_label}] time 'add' args must be >= 1. Offender in '{pt.get('name')}'.")
            if fname == "duration":
                for tr in dim.get("transformations", []):
                    op = (tr.get("name") or "").lower()
                    if op in {"mul", "div"}:
                        a = int(tr.get("args", [0])[0])
                        if a in (0, 1):
                            raise RuntimeError(f"[{bar_label}] duration mul/div args must be 2..4. Offender in '{pt.get('name')}'.")

# ---------- main API ----------
def generate_unit_from_template(
    template_path: pathlib.Path,
    *,
    label: Optional[str] = None,
    default_bar_ticks: int = 12,      # default grid if your template doesn't specify
    fetch: bool = False,
    acct: Optional[Account] = None,
) -> Dict[str, Any]:
    """
    Generate ONE UNIT (one prompt file). The unit can contain many bars.

    Returns a dict:
      {
        "unit_label": <str>,
        "run_dir": <Path>,
        "total_ticks": <int>,
        "schedule": [ {bar_index, start_tick, actual_end_tick, slot_length_tick, ...}, ... ],
        "per_instr": { instrument: { "time":[...], "pitch":[...], ... } },
        "payload": <visualiser_payload>
      }
    """
    label = label or template_path.stem
    run_dir = _make_run_dir(label=label)

    # --- DCN auth/account -----------------------------------------------------
    if acct is None:
        priv = os.getenv("PRIVATE_KEY")
        acct = Account.from_key(priv) if priv else Account.create("TEMPKEY for demo")
    print(f"[{label}] Using account: {acct.address}")

    dcn = DCNClient(API_BASE)
    nonce = dcn.get_nonce(acct.address)
    msg   = f"Login nonce: {nonce}"
    sig   = acct.sign_message(encode_defunct(text=msg)).signature.hex()
    dcn.post_auth(acct.address, msg, sig)
    if not (dcn.access_token and dcn.refresh_token):
        raise RuntimeError("Auth failed — missing tokens")

    # --- Meter/ticks for this unit (uniform per unit; can be extended to per-bar) ---
    bar_ticks = int(default_bar_ticks)
    NUM, DEN  = meter_from_ticks(bar_ticks)

    # --- Prompts ---------------------------------------------------------------
    system_text = load_system_prompt(fallback="")

    sidecar_vars_path = template_path.with_suffix(template_path.suffix + ".vars.json")
    extra_vars = None
    if sidecar_vars_path.exists():
        try:
            extra_vars = json.loads(sidecar_vars_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to read sidecar vars: {sidecar_vars_path}: {e}")
        
    user_text   = render_user_prompt_from_file(
        template_path,
        bar_ticks=bar_ticks,
        num=NUM, den=DEN,
        instruments=INSTRUMENTS,
        extra_vars=extra_vars,
    )

    _save_text(run_dir / "01_system_prompt.txt", system_text)
    _save_text(run_dir / "02_user_prompt.rendered.json", user_text)

    # --- OpenAI call (Responses + GPT-5) --------------------------------------
    OPENAI_API_KEY = _load_openai_key()
    oai = OpenAI(api_key=OPENAI_API_KEY)

    resp = oai.responses.create(
        model="gpt-5",
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
            {"role": "user",   "content": [{"type": "input_text", "text": user_text}]}
        ],
        temperature=1,
        text={"format": {"type": "json_object"}},
    )

    raw_text = (resp.output_text or "").strip()
    _save_text(run_dir / "03_llm_output_raw.json", raw_text)
    obj = json.loads(raw_text)
    _save_json(run_dir / "04_llm_output_parsed.json", obj)

    # Determine if SINGLE bar bundle or SECTION with many bars
    if "bars" in obj and isinstance(obj["bars"], list):
        bar_bundles = obj["bars"]
    else:
        bar_bundles = [obj]  # legacy single-bar

    # Unit-level accumulators
    per_instr_unit = {i: {"pitch": [], "time": [], "duration": [], "velocity": [], "numerator": [], "denominator": []}
                      for i in ORDERED_INSTRS}
    schedule: List[Dict[str, Any]] = []
    cumulative = 0

    # Process each returned bar bundle
    for local_idx, bundle in enumerate(bar_bundles, start=1):
        bar_label = f"bar{local_idx:02d}"

        # Validate
        _validate_bundle(bundle, bar_label)

        # Unique renaming + run_plan consistency
        name_map: Dict[str, str] = {}; seen = set()
        for feat in bundle.get("features", []):
            meta = feat.get("meta", {})
            instr = meta.get("instrument", "unknown")
            bar   = meta.get("bar", local_idx)
            role  = meta.get("role", "feat")
            old   = feat["pt"].get("name", f"{instr}_b{bar}_{role}")
            new   = _make_unique_name(instr, bar, role, seen)
            name_map[old] = new
            feat["pt"]["name"] = new

        for rp in bundle.get("run_plan", []):
            old = rp.get("feature_name","")
            if old in name_map:
                rp["feature_name"] = name_map[old]

        bundle["created_feature_names"] = [f["pt"]["name"] for f in bundle.get("features", [])]

        # Enforce meter seeds + constant transforms
        for rp in bundle.get("run_plan", []):
            rp.setdefault("seeds", {})
            rp["seeds"]["numerator"]   = NUM
            rp["seeds"]["denominator"] = DEN
        for feat in bundle.get("features", []):
            for d in feat.get("pt", {}).get("dimensions", []):
                fn = (d.get("feature_name") or "").strip().lower()
                if fn in ("numerator", "denominator"):
                    d["transformations"] = [{"name":"add","args":[0]}]

        # Save per-bar bundle artifacts
        _save_json(run_dir / f"05_{bar_label}_name_map.json", name_map)
        _save_json(run_dir / f"06_{bar_label}_bundle_final_to_post.json", bundle)
        _save_json(run_dir / f"07_{bar_label}_run_plan.json", bundle.get("run_plan", []))

        # Register PTs + receipts
        receipts: List[Dict[str, Any]] = []
        for feat in bundle.get("features", []):
            res = dcn.post_feature(feat["pt"])
            receipts.append({"pt_name": feat["pt"]["name"], "response": res})
        _save_json(run_dir / f"08_{bar_label}_post_feature_receipts.json", receipts)

        # Execute and collect
        dims_by_name = {feat["pt"]["name"]: feat["pt"]["dimensions"] for feat in bundle.get("features", [])}
        exec_by_feature: Dict[str, Dict[str, List[int]]] = {}

        for rp in bundle.get("run_plan", []):
            fname = rp["feature_name"]; dims = dims_by_name.get(fname, [])
            seeds = {k:int(v) for (k,v) in (rp.get("seeds") or {}).items()}
            N     = int(rp.get("N", 4))
            samples = dcn.execute_pt(acct, fname, N, seeds, dims)
            streams = _normalize_exec(samples)
            exec_by_feature[fname] = streams

            # route to instrument
            instr = next((f["meta"].get("instrument") for f in bundle.get("features", []) if f["pt"]["name"] == fname), None)
            if not instr: 
                continue
            # NOTE: do not offset yet; first measure actual_end
            pass

        _save_json(run_dir / f"09_{bar_label}_exec_by_feature.json", exec_by_feature)

        # Build a per-bar instrument map for measuring length and later offsetting
        per_instr_bar = {i: {"pitch": [], "time": [], "duration": [], "velocity": []} for i in ORDERED_INSTRS}
        for fname, streams in exec_by_feature.items():
            instr = next((f["meta"].get("instrument") for f in bundle.get("features", []) if f["pt"]["name"] == fname), None)
            if not instr:
                continue
            for key in ("pitch","time","duration","velocity"):
                if key in streams:
                    per_instr_bar[instr][key].extend(list(streams[key]))

        # Measure bar's actual end
        actual_end = 0
        for instr in ORDERED_INSTRS:
            times = per_instr_bar[instr]["time"]
            durs  = per_instr_bar[instr]["duration"]
            if times:
                seg_end = max(t + d for t, d in zip(times, durs))
                if seg_end > actual_end:
                    actual_end = seg_end

        # Slot length is at least one bar grid
        slot_length = max(actual_end, bar_ticks)
        schedule.append({
            "bar_local_index": local_idx,
            "start_tick": cumulative,
            "actual_end_tick": actual_end,
            "slot_length_tick": slot_length,
            "bar_ticks": bar_ticks,
            "meter": {"numerator": NUM, "denominator": DEN},
        })

        # Append into UNIT with offset
        for instr in ORDERED_INSTRS:
            src = per_instr_bar[instr]
            # offset times
            per_instr_unit[instr]["time"].extend([t + cumulative for t in src["time"]])
            # copy other streams
            for key in ("pitch","duration","velocity"):
                per_instr_unit[instr][key].extend(list(src[key]))
            # meter arrays same length as *new* time
            L = len(src["time"])
            per_instr_unit[instr]["numerator"].extend([NUM] * L)
            per_instr_unit[instr]["denominator"].extend([DEN] * L)

        cumulative += slot_length

    # Build unit-level visualiser payload
    tracks = {instr: _instrument_pack(instr, per_instr_unit[instr]) for instr in ORDERED_INSTRS}
    payload = {"instrument_meta": INSTRUMENT_META, "tracks": tracks}

    # Persist unit-level payload + schedule + manifest
    _save_json(run_dir / "10_unit_visualiser_payload.json", payload)
    _save_json(run_dir / "11_unit_schedule.json", schedule)
    manifest = {
        "unit_label": label,
        "template": str(template_path),
        "total_ticks": cumulative,
        "bars_count": len(schedule),
        "files": {
            "system_prompt": "01_system_prompt.txt",
            "user_prompt_rendered": "02_user_prompt.rendered.json",
            "llm_raw": "03_llm_output_raw.json",
            "llm_parsed": "04_llm_output_parsed.json",
            "per_bar_bundles": "06_*_bundle_final_to_post.json",
            "per_bar_exec": "09_*_exec_by_feature.json",
            "unit_payload": "10_unit_visualiser_payload.json",
            "unit_schedule": "11_unit_schedule.json",
        }
    }
    _save_json(run_dir / "manifest.json", manifest)

    # Convenience copy at project root (optional)
    root_payload = pathlib.Path(__file__).resolve().parent / f"{label}_unit.json"
    with open(root_payload, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[{label}] Unit total ticks: {cumulative}")
    print(f"[{label}] Run dir: {run_dir}")
    print(f"[{label}] Wrote {root_payload.name}")

    return {
        "unit_label": label,
        "run_dir": run_dir,
        "total_ticks": cumulative,
        "schedule": schedule,
        "per_instr": per_instr_unit,
        "payload": payload,
    }
