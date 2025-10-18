#!/usr/bin/env python3
"""
compose_suite.py — discover prompts/user/*.txt (and *.template.json),
generate each as a UNIT (one prompt → 1..N bars),
feed a rolling CONTEXT to each subsequent unit,
concatenate units into a final composition, and
write EVERYTHING into a single suite folder under runs/.
"""

import json, pathlib
from typing import Dict, List, Any
from datetime import datetime

from pt_config import ORDERED_INSTRS, INSTRUMENT_META
from pt_prompts import PROMPTS_DIR
from pt_generate import generate_unit_from_template

def _discover_templates() -> List[pathlib.Path]:
    user_dir = PROMPTS_DIR / "user"
    if not user_dir.exists():
        raise SystemExit("Missing prompts/user/ directory.")
    paths = list(user_dir.glob("*.template.json")) + list(user_dir.glob("*.txt"))
    if not paths:
        raise SystemExit("No prompt templates found in prompts/user/ (expected *.txt or *.template.json).")
    return sorted(paths, key=lambda p: p.name)

def _concat_units(units: List[Dict[str, Any]]) -> Dict[str, Any]:
    combined: Dict[str, Dict[str, List[int]]] = {
        instr: {k: [] for k in ("pitch","time","duration","velocity","numerator","denominator")}
        for instr in ORDERED_INSTRS
    }
    suite_schedule: List[Dict[str, Any]] = []
    cumulative = 0

    for u in units:
        per_instr = u["per_instr"]; total = int(u["total_ticks"])
        suite_schedule.append({
            "unit_label": u["unit_label"],
            "start_tick": cumulative,
            "unit_ticks": total
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

    def _pack(instrument: str, streams: Dict[str, List[int]]) -> List[dict]:
        return [{"feature_path": f"/{instrument}/{scalar}", "data": list(streams[scalar])}
                for scalar in ("pitch","time","duration","velocity","numerator","denominator")]

    tracks = {instr: _pack(instr, combined[instr]) for instr in ORDERED_INSTRS}
    payload = {"instrument_meta": INSTRUMENT_META, "tracks": tracks}
    return {"payload": payload, "schedule": suite_schedule, "total_ticks": cumulative}

def main():
    templates = _discover_templates()
    print("Discovered templates (units):")
    for p in templates:
        print("  •", p.name)

    # Create ONE suite folder upfront
    runs_root = pathlib.Path(__file__).resolve().parent / "runs"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suite_dir = runs_root / f"{ts}_suite"
    suite_dir.mkdir(parents=True, exist_ok=False)

    units: List[Dict[str, Any]] = []
    suite_context = ""         # rolling text summary passed to each next unit
    suite_pt_journal: List[Dict[str, Any]] = []  # merged journal across units
    prompts_log_lines: List[str] = []            # optional: keep what we asked

    for path in templates:
        unit = generate_unit_from_template(path, label=path.stem, fetch=False, suite_context=suite_context)


        units.append(unit)

        # Merge PT journal
        suite_pt_journal.extend(unit.get("pt_journal", []))

        # Grow context with the rendered prompt and the computed summary
        prompts_log_lines.append(f"PROMPT {unit['unit_label']}:\n{unit['rendered_user_text']}\n")
        prompts_log_lines.append(f"SUMMARY {unit['unit_label']}:\n{unit['unit_summary_text']}\n")
        suite_context = (suite_context + "\n\n" +
                         f"PROMPT {unit['unit_label']}:\n{unit['rendered_user_text']}\n\n" +
                         f"SUMMARY {unit['unit_label']}:\n{unit['unit_summary_text']}\n").strip()

    # Concatenate units
    suite = _concat_units(units)

    # Persist EVERYTHING (once) in the single suite folder
    with open(suite_dir / "composition_suite.json", "w", encoding="utf-8") as f:
        json.dump(suite["payload"], f, ensure_ascii=False, indent=2)
    with open(suite_dir / "schedule.json", "w", encoding="utf-8") as f:
        json.dump(suite["schedule"], f, ensure_ascii=False, indent=2)
    with open(suite_dir / "pt_journal.json", "w", encoding="utf-8") as f:
        json.dump(suite_pt_journal, f, ensure_ascii=False, indent=2)
    with open(suite_dir / "prompts_and_summaries.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(prompts_log_lines))

    manifest = {
        "units": [u["unit_label"] for u in units],
        "total_ticks": suite["total_ticks"],
        "files": {
            "payload": "composition_suite.json",
            "schedule": "schedule.json",
            "pt_journal": "pt_journal.json",
            "prompts_and_summaries": "prompts_and_summaries.txt"
        }
    }
    with open(suite_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Suite written:")
    print("  •", suite_dir / "composition_suite.json")
    print("  •", suite_dir / "schedule.json")
    print("  •", suite_dir / "pt_journal.json")
    print("  •", suite_dir / "manifest.json")

if __name__ == "__main__":
    main()
