#!/usr/bin/env python3
"""
compose_suite.py — discovers prompts/user/*.template.json,
generates each as a UNIT (one prompt → 1..N bars),
then concatenates units back-to-back into a final composition payload.
"""

import json, pathlib
from typing import Dict, List, Any
from datetime import datetime

from pt_config import ORDERED_INSTRS, INSTRUMENT_META
from pt_prompts import PROMPTS_DIR
from pt_generate import generate_unit_from_template

# ---------- discovery ----------
def _discover_templates() -> List[pathlib.Path]:
    """
    Find all prompts/user/*.template.json and return lexicographically sorted paths.
    Name your files 001_intro.template.json, 010_sectionA.template.json, etc. for ordering.
    """
    user_dir = PROMPTS_DIR / "user"
    if not user_dir.exists():
        raise SystemExit("Missing prompts/user/ directory.")
    paths = sorted(user_dir.glob("*.template.json"))
    if not paths:
        raise SystemExit("No prompt templates found in prompts/user/.")
    return paths

# ---------- concatenation of UNITS ----------
def _concat_units(units: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Each UNIT already concatenated its own bars.
    We only need to offset between units and glue arrays.
    """
    combined: Dict[str, Dict[str, List[int]]] = {
        instr: {k: [] for k in ("pitch","time","duration","velocity","numerator","denominator")}
        for instr in ORDERED_INSTRS
    }
    suite_schedule: List[Dict[str, Any]] = []
    cumulative = 0

    for u in units:
        per_instr = u["per_instr"]
        total = int(u["total_ticks"])
        suite_schedule.append({
            "unit_label": u["unit_label"],
            "start_tick": cumulative,
            "unit_ticks": total,
            "run_dir": str(u["run_dir"]),
        })

        for instr in ORDERED_INSTRS:
            src = per_instr[instr]
            combined[instr]["time"].extend([t + cumulative for t in src["time"]])
            for key in ("pitch","duration","velocity"):
                combined[instr][key].extend(list(src[key]))
            L = len(src["time"])
            combined[instr]["numerator"].extend(src["numerator"][:L])
            combined[instr]["denominator"].extend(src["denominator"][:L])

        cumulative += total

    # Build final payload
    def _pack(instrument: str, streams: Dict[str, List[int]]) -> List[dict]:
        out = []
        for scalar in ("pitch","time","duration","velocity","numerator","denominator"):
            out.append({"feature_path": f"/{instrument}/{scalar}", "data": list(streams[scalar])})
        return out

    tracks = {instr: _pack(instr, combined[instr]) for instr in ORDERED_INSTRS}
    payload = {"instrument_meta": INSTRUMENT_META, "tracks": tracks}
    return {"payload": payload, "schedule": suite_schedule, "total_ticks": cumulative}

# ---------- main ----------
def main():
    templates = _discover_templates()
    print("Discovered templates (units):")
    for p in templates:
        print("  •", p.name)

    # Generate each unit
    units: List[Dict[str, Any]] = []
    for path in templates:
        unit = generate_unit_from_template(path, label=path.stem, fetch=False)
        units.append(unit)

    # Concatenate units
    suite = _concat_units(units)

    # Persist the suite
    runs_root = pathlib.Path(__file__).resolve().parent / "runs"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suite_dir = runs_root / f"{ts}_suite"
    suite_dir.mkdir(parents=True, exist_ok=False)

    with open(suite_dir / "composition_suite.json", "w", encoding="utf-8") as f:
        json.dump(suite["payload"], f, ensure_ascii=False, indent=2)
    with open(suite_dir / "schedule.json", "w", encoding="utf-8") as f:
        json.dump(suite["schedule"], f, ensure_ascii=False, indent=2)

    manifest = {
        "units": [u["unit_label"] for u in units],
        "total_ticks": suite["total_ticks"],
        "files": {"payload": "composition_suite.json", "schedule": "schedule.json"}
    }
    with open(suite_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Convenience root copy
    root_out = pathlib.Path(__file__).resolve().parent / "composition_suite.json"
    with open(root_out, "w", encoding="utf-8") as f:
        json.dump(suite["payload"], f, ensure_ascii=False, indent=2)

    print("Suite written:")
    print("  •", suite_dir / "composition_suite.json")
    print("  •", suite_dir / "schedule.json")
    print("  •", suite_dir / "manifest.json")
    print("Root copy:")
    print("  •", root_out)

if __name__ == "__main__":
    main()
