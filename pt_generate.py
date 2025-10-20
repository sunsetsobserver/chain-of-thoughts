"""
pt_generate.py — generate ONE UNIT (one prompt file) using GPT-5 + DCN.
The prompt may yield a SINGLE bar bundle or a SECTION with many bars:
- If the model returns {"features":..., "run_plan":...}: one bar.
- If it returns {"bars":[ {...}, {...}, ... ]}: multi-bar section.

Supports "context chaining": pass `suite_context` (text of prior bundles)
as an extra system message so the model composes with awareness of what came before.
"""

import os, json, time, pathlib, importlib.util, statistics
from typing import Dict, List, Any, Optional
from datetime import datetime
from eth_account import Account
from openai import OpenAI, APITimeoutError

from pt_config import (
    API_BASE, ORDERED_INSTRS, INSTRUMENTS, INSTRUMENT_META,
    BAR_TICKS_BY_BAR, meter_from_ticks
)
from pt_prompts import load_system_prompt, render_user_prompt_file, PROMPTS_DIR
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

def _cap_to_next_onset_and_bar(times: List[int], durations: List[int], bar_ticks: int) -> tuple[list[int], list[int]]:
    """
    Make durations safe:
      - For each note i, cap to next onset (or bar end for the last note)
      - Cap to bar end (<= bar_ticks)
      - Drop zero/negative-length notes after capping
    Assumes times are strictly increasing (DCN/PT ensures this after validation).
    """
    if not times:
        return [], []
    T = list(times)
    D = list(durations)
    n = len(T)
    for i in range(n):
        t = T[i]
        nxt = T[i + 1] if i + 1 < n else bar_ticks
        max_ok = max(0, nxt - t)
        if D[i] > max_ok:
            D[i] = max_ok
        if t + D[i] > bar_ticks:
            D[i] = max(0, bar_ticks - t)
    keep_idx = [i for i in range(n) if D[i] > 0 and T[i] < bar_ticks]
    return [T[i] for i in keep_idx], [D[i] for i in keep_idx]


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
def _must_uint(x, label: str) -> int:
    if not isinstance(x, int):
        raise RuntimeError(f"{label} must be an unsigned integer")
    if x < 0:
        raise RuntimeError(f"{label} must be unsigned (>= 0)")
    return x

# ---------- context summary ----------
def _summarize_unit(per_instr_unit: Dict[str, Dict[str, List[int]]],
                    bars_count: int, total_ticks: int, num: int, den: int,
                    label: str) -> str:
    lines = [f"UNIT {label}: bars={bars_count}, total_ticks={total_ticks}, meter={num}/{den}"]
    for instr in ORDERED_INSTRS:
        s = per_instr_unit[instr]
        notes = len(s["time"])
        if notes == 0:
            lines.append(f"- {instr}: notes=0")
            continue
        ps = s["pitch"]; vs = s["velocity"]; ts = s["time"]; ds = s["duration"]
        pitch_min, pitch_max = min(ps), max(ps)
        avg_vel = round(sum(vs) / len(vs), 1) if vs else 0
        last_pitch = ps[-1]; last_on = ts[-1]; last_end = ts[-1] + ds[-1]
        lines.append(
            f"- {instr}: notes={notes}, pitch[{pitch_min}..{pitch_max}], "
            f"avg_vel={avg_vel}, last_pitch={last_pitch}, last_on={last_on}, last_end={last_end}"
        )
    return "\n".join(lines)

# ---------- JSON extraction helpers ----------
def _resp_to_dict_safe(resp) -> dict:
    try:
        # SDK objects typically support model_dump / to_dict
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        if hasattr(resp, "to_dict"):
            return resp.to_dict()
    except Exception:
        pass
    # very defensive fallback
    try:
        return json.loads(getattr(resp, "model_dump_json", lambda: "{}")())
    except Exception:
        return {}

def _extract_output_text_parts(resp) -> List[str]:
    parts: List[str] = []
    try:
        for msg in getattr(resp, "output", []) or []:
            for c in getattr(msg, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    txt = getattr(c, "text", "") or ""
                    if txt:
                        parts.append(txt)
    except Exception:
        pass
    return parts

def _first_json_block(text: str) -> str:
    """Heuristic: slice the first {...} block if the model wrapped JSON in prose."""
    if not text:
        return ""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return ""

# ---------- main API ----------
def generate_unit_from_template(
    template_path: pathlib.Path,
    *,
    label: Optional[str] = None,
    default_bar_ticks: int = 12,
    fetch: bool = False,
    acct: Optional[Account] = None,
    suite_context: Optional[str] = None,   # rolling context string
    session_dir: Optional[pathlib.Path] = None,  # shared suite folder (for error dumps)
    dcn: Optional[DCNClient] = None,
) -> Dict[str, Any]:
    """
    Generate ONE UNIT (one prompt file). The unit can contain many bars.
    Returns dict with:
      unit_label, total_ticks, schedule, per_instr, payload,
      rendered_user_text, unit_summary_text, pt_journal, model_bundle_minjson
    """
    label = label or template_path.stem

    # --- DCN auth/account -----------------------------------------------------
    if acct is None:
        priv = os.getenv("PRIVATE_KEY")
        acct = Account.from_key(priv) if priv else Account.create("TEMPKEY for demo")
    print(f"[{label}] Using account: {acct.address}")

    # Reuse DCN client if provided; otherwise create and auth once here
    dcn = dcn or DCNClient(API_BASE)
    dcn.ensure_auth(acct)

    from pt_prompts import parse_prompt_directives

    # --- Meter/ticks for this unit (prompt overrides allowed) -----------------
    overrides = parse_prompt_directives(template_path)
    if "bar_ticks" in overrides:
        bar_ticks = int(overrides["bar_ticks"])
        NUM, DEN = meter_from_ticks(bar_ticks)
    elif "num" in overrides and "den" in overrides:
        NUM = int(overrides["num"]); DEN = int(overrides["den"])
        bar_ticks = int((16 * NUM) // DEN)  # 1 tick = 1/16 note
    else:
        bar_ticks = int(default_bar_ticks)
        NUM, DEN  = meter_from_ticks(bar_ticks)

    # --- Prompts ---------------------------------------------------------------
    system_text = load_system_prompt(fallback="")
    user_text   = render_user_prompt_file(
        template_path,
        bar_ticks=bar_ticks,
        num=NUM, den=DEN,
        instruments=INSTRUMENTS,
    )

    # --- OpenAI call (Responses API + Structured Outputs) ---------------------
    OPENAI_API_KEY = _load_openai_key()
    oai = OpenAI(
        api_key=OPENAI_API_KEY
    )

    # Messages (with optional rolling context as extra system message)
    messages = [
        {"role": "system", "content": [{"type": "input_text", "text": system_text}]}
    ]
    if suite_context:
        messages.append({
            "role": "system",
            "content": [{
                "type": "input_text",
                "text": (
                    "REFERENCE PT JSON RESPONSES FROM EARLIER UNITS IF NEEDED:\n"
                    "Use these ONLY for continuity or literal reuse of earlier fragments.\n"
                    "Do NOT quote, summarize, or echo these objects.\n"
                    "Output exactly ONE JSON object for the CURRENT unit only.\n\n"
                    + suite_context
                )
            }]
        })
    messages.append({
        "role": "user",
        "content": [{"type": "input_text", "text": user_text}]
    })

    oai = OpenAI(api_key=OPENAI_API_KEY, max_retries=2)  # SDK will retry briefly on 5xx

    resp = oai.responses.create(
        model="gpt-5",
        input=messages,
        text={"format": {"type": "json_object"}},   # flexible JSON mode
        reasoning={"effort": "low"},                # keep internal chains short
    )
    raw_text = (resp.output_text or "").strip()

    obj = json.loads(raw_text)
    bar_bundles = (
        obj["bars"] if isinstance(obj, dict) and isinstance(obj.get("bars"), list)
        else [obj]
    )

    model_bundle_minjson = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    print(f"[{label}] Parsed model JSON: bars={len(bar_bundles)}")

    # Unit accumulators
    per_instr_unit = {i: {"pitch": [], "time": [], "duration": [], "velocity": [], "numerator": [], "denominator": []}
                      for i in ORDERED_INSTRS}
    schedule: List[Dict[str, Any]] = []
    cumulative = 0

    # Minimal per-unit journal of created/posted/executed PTs
    pt_journal: List[Dict[str, Any]] = []

    # Process each returned bar bundle
    for local_idx, bundle in enumerate(bar_bundles, start=1):
        bar_label = f"bar{local_idx:02d}"

        # Rename features uniquely + fix run_plan refs
        name_map: Dict[str, str] = {}; seen = set()
        feats = bundle.get("features", [])
        runp = bundle.get("run_plan", [])
        print(f"[{label}] {bar_label}: features={len(feats)}, run_plan={len(runp)}")

        for feat in feats:
            meta = feat.get("meta", {})
            instr = meta.get("instrument", "unknown")
            bar   = meta.get("bar", local_idx)
            role  = meta.get("role", "feat")
            old   = feat["pt"].get("name", f"{instr}_b{bar}_{role}")
            new   = _make_unique_name(instr, bar, role, seen)
            name_map[old] = new
            feat["pt"]["name"] = new
        for rp in runp:
            old = rp.get("feature_name","")
            if old in name_map:
                rp["feature_name"] = name_map[old]
        bundle["created_feature_names"] = [f["pt"]["name"] for f in feats]

        # Enforce meter seeds + constant transforms
        for rp in runp:
            rp.setdefault("seeds", {})
            rp["seeds"]["numerator"]   = NUM
            rp["seeds"]["denominator"] = DEN
        for feat in feats:
            for d in feat.get("pt", {}).get("dimensions", []):
                fn = (d.get("feature_name") or "").strip().lower()
                if fn in ("numerator", "denominator"):
                    d["transformations"] = [{"name":"add","args":[0]}]

        # Register PTs
        for i, feat in enumerate(feats, start=1):
            pt_name = feat["pt"]["name"]
            instr = (feat.get("meta") or {}).get("instrument", "")
            print(f"[{label}] {bar_label}: POST feature {i}/{len(feats)} name={pt_name} instr={instr}")
            try:
                res = dcn.post_feature(feat["pt"], acct=acct)
            except Exception as e:
                if session_dir:
                    errdir = session_dir / "errors"
                    errdir.mkdir(parents=True, exist_ok=True)
                    _save_json(errdir / f"{label}.{bar_label}.feature_{i}.{instr}.json", feat["pt"])
                    _save_text(errdir / f"{label}.{bar_label}.feature_{i}.{instr}.error.txt", repr(e))
                    # If requests.HTTPError, capture server body:
                    try:
                        import requests
                        if isinstance(e, requests.HTTPError) and e.response is not None:
                            _save_text(errdir / f"{label}.{bar_label}.feature_{i}.{instr}.server.txt", e.response.text)
                    except Exception:
                        pass
                raise


        # Execute and collect
        dims_by_name = {feat["pt"]["name"]: feat["pt"]["dimensions"] for feat in feats}
        exec_by_feature: Dict[str, Dict[str, List[int]]] = {}

        for rp in runp:
            fname = rp["feature_name"]
            dims = dims_by_name.get(fname, [])
            seeds = {k:int(v) for (k,v) in (rp.get("seeds") or {}).items()}
            N = int(rp.get("N", 4))

            print(f"[{label}] {bar_label}: EXEC {fname} N={N}")
            samples = dcn.execute_pt(acct, fname, N, seeds, dims)
            streams = _normalize_exec(samples)
            exec_by_feature[fname] = streams

            # Journal entry for execution (counts only)
            pt_journal.append({
                "action": "execute_pt",
                "unit": label,
                "bar_index": local_idx,
                "pt_name": fname,
                "N": N,
                "seeds": seeds,
                "sample_counts": {k: len(v) for k, v in streams.items()}
            })

        # Build per-bar instrument map (to compute actual_end, then offset)
        per_instr_bar = {i: {"pitch": [], "time": [], "duration": [], "velocity": []} for i in ORDERED_INSTRS}
        for fname, streams in exec_by_feature.items():
            instr = next((f["meta"].get("instrument") for f in feats if f["pt"]["name"] == fname), None)
            if not instr:
                continue
            for key in ("pitch","time","duration","velocity"):
                if key in streams:
                    per_instr_bar[instr][key].extend(list(streams[key]))
        
        # Cap durations to next onset and to the bar boundary
        for instr in ORDERED_INSTRS:
            t = per_instr_bar[instr]["time"]
            d = per_instr_bar[instr]["duration"]
            if t and d:
                t_cap, d_cap = _cap_to_next_onset_and_bar(t, d, bar_ticks)
                per_instr_bar[instr]["time"] = t_cap
                per_instr_bar[instr]["duration"] = d_cap

        # Measure actual end AFTER capping
        actual_end = 0
        for instr in ORDERED_INSTRS:
            times = per_instr_bar[instr]["time"]
            durs  = per_instr_bar[instr]["duration"]
            if times and durs:
                seg_end = max(t + d for t, d in zip(times, durs))
                if seg_end > actual_end:
                    actual_end = seg_end

        # Force fixed bar slot; material is guaranteed to fit now
        slot_length = bar_ticks

        # Append into UNIT with offset
        print(f"[{label}] {bar_label}: end={actual_end} / slot_length={slot_length}")
        schedule.append({
            "bar_local_index": local_idx,
            "start_tick": cumulative,
            "actual_end_tick": actual_end,
            "slot_length_tick": slot_length,
            "bar_ticks": bar_ticks,
            "meter": {"numerator": NUM, "denominator": DEN},
        })
        for instr in ORDERED_INSTRS:
            src = per_instr_bar[instr]
            per_instr_unit[instr]["time"].extend([t + cumulative for t in src["time"]])
            for key in ("pitch","duration","velocity"):
                per_instr_unit[instr][key].extend(list(src[key]))
            L = len(src["time"])
            per_instr_unit[instr]["numerator"].extend([NUM] * L)
            per_instr_unit[instr]["denominator"].extend([DEN] * L)

        cumulative += slot_length

    # Build unit-level visualiser payload (returned to caller)
    tracks = {instr: _instrument_pack(instr, per_instr_unit[instr]) for instr in ORDERED_INSTRS}
    payload = {"instrument_meta": INSTRUMENT_META, "tracks": tracks}

    # Human-readable summary for context chaining (returned; not written here)
    unit_summary_text = _summarize_unit(per_instr_unit, bars_count=len(schedule), total_ticks=cumulative,
                                        num=NUM, den=DEN, label=label)

    print(f"[{label}] Unit total ticks: {cumulative}")

    return {
        "unit_label": label,
        "total_ticks": cumulative,
        "schedule": schedule,
        "per_instr": per_instr_unit,
        "payload": payload,
        "rendered_user_text": user_text,
        "unit_summary_text": unit_summary_text,
        "pt_journal": pt_journal,
        "model_bundle_minjson": model_bundle_minjson,
    }
