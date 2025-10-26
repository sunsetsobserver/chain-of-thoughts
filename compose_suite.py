#!/usr/bin/env python3
"""
compose_suite.py — discover prompts/user/*.txt (and *.template.json),
generate each as a UNIT (one prompt → 1..N bars),
feed a rolling CONTEXT to each subsequent unit,
concatenate units into a final composition, and
write EVERYTHING into a single suite folder under runs/.

Additions:
- Checkpointing after each unit + periodic partial outputs.
- Resume support: --resume runs/<ts>_suite.
- Rich logging to console and suite.log.
- Honors ONLY filter and context budget you already added.
"""

import argparse
import json, pathlib
from typing import Dict, List, Any
from datetime import datetime

from pt_config import ORDERED_INSTRS, INSTRUMENT_META
from pt_prompts import PROMPTS_DIR
from pt_generate import generate_unit_from_template

from eth_account import Account
from dcn_client import DCNClient
from pt_config import API_BASE
import os
import subprocess
import sys
import logging
import time

from tqdm import tqdm


def _mk_bundle_context(
    snippets: List[str],
    budget_chars: int = 15000,
    max_items: int | None = None,
    newest_first: bool = False,
    separator: str = "\n\n-----\n\n",
) -> str:
    """
    Build the rolling system context from prior model JSON bundles.
    - max_items=None  → use all snippets (default)
      max_items>=0    → use only the last N snippets (0 = none)
    - budget_chars caps total characters.
    - newest_first controls final ordering for readability (doesn't affect model ability).
    """
    if not snippets:
        return ""
    # Take only the last N if requested
    seq = snippets[-max_items:] if (isinstance(max_items, int) and max_items >= 0) else list(snippets)
    # Pack from newest to oldest until budget is hit
    packed, total = [], 0
    for s in reversed(seq):
        need = len(s) + (len(separator) if packed else 0)
        if total + need > budget_chars:
            break
        if packed:
            packed.append(separator)
        packed.append(s)
        total += need
    # packed is newest→older; flip if caller prefers oldest→newest
    final = "".join(reversed(packed)) if not newest_first else "".join(packed)
    return final


def _discover_templates() -> List[pathlib.Path]:
    user_dir = PROMPTS_DIR / "user"
    if not user_dir.exists():
        raise SystemExit("Missing prompts/user/ directory.")
    paths = list(user_dir.glob("*.template.json")) + list(user_dir.glob("*.txt"))
    if not paths:
        raise SystemExit("No prompt templates found in prompts/user/ (expected *.txt or *.template.json).")
    paths = sorted(paths, key=lambda p: p.name)
    only = os.getenv("ONLY")  # e.g., ONLY=010.txt or ONLY=010*
    if only:
        import fnmatch
        paths = [p for p in paths if fnmatch.fnmatch(p.name, only)]
        if not paths:
            raise SystemExit(f"ONLY filter '{only}' matched no files.")
    return paths


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


# ---------- checkpoint helpers ----------
def _write_json(path: pathlib.Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _append_text(path: pathlib.Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def _load_json(path: pathlib.Path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_checkpoint(suite_dir: pathlib.Path,
                     templates: List[pathlib.Path],
                     units_done: List[Dict[str, Any]],
                     ctx_bundles: List[str],
                     checkpoint_every: int,
                     force_partial: bool = False):
    """
    Persist durable state so we can resume later.
    - checkpoint.json: progress + template order + ctx bundles (minjson)
    - units/<label>.json: full unit objects
    - composition_suite.partial.json & schedule.partial.json every K units (or on force)
    """
    ckpt = {
        "templates": [str(p) for p in templates],
        "done_units": [u["unit_label"] for u in units_done],
        "ctx_bundles": list(ctx_bundles),
        "progress": {"completed": len(units_done), "total": len(templates)},
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(suite_dir / "checkpoint.json", ckpt)

    # Per-unit files (idempotent)
    units_dir = suite_dir / "units"
    for u in units_done:
        _write_json(units_dir / f"{u['unit_label']}.json", u)

    # Partial stitched outputs
    if force_partial or (len(units_done) % max(1, checkpoint_every) == 0):
        stitched = _concat_units(units_done)
        _write_json(suite_dir / "composition_suite.partial.json", stitched["payload"])
        _write_json(suite_dir / "schedule.partial.json", stitched["schedule"])
        logging.info("Wrote partial outputs (%d/%d).",
                     len(units_done), len(templates))


# ---------- logging ----------
def _setup_logging(suite_dir: pathlib.Path, verbose: bool):
    log_fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=log_fmt, handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(suite_dir / "suite.log", encoding="utf-8"),
    ])
    logging.info("Logging to %s", suite_dir / "suite.log")


def _maybe_export_midi(suite_dir: pathlib.Path):
    """
    Try to export a MIDI file using Node tools/pt2midi.js.
    Non-fatal on any failure. Set NO_MIDI=1 to skip.
    """
    if os.getenv("NO_MIDI", "0") == "1":
        logging.info("MIDI export skipped (NO_MIDI=1).")
        return

    tools_js = pathlib.Path(__file__).resolve().parent / "tools" / "pt2midi.js"
    if not tools_js.exists():
        logging.info("MIDI export skipped (tools/pt2midi.js not found).")
        return

    in_json = suite_dir / "composition_suite.json"
    out_mid = suite_dir / "composition_suite.mid"

    try:
        result = subprocess.run(
            ["node", str(tools_js), str(in_json), str(out_mid)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.stdout.strip():
            logging.info(result.stdout.strip())
        if out_mid.exists():
            logging.info("  • %s", out_mid)
        else:
            logging.warning("MIDI export completed but output file was not found where expected.")
    except FileNotFoundError:
        logging.info("MIDI export skipped (Node not found on PATH).")
    except subprocess.CalledProcessError as e:
        logging.warning("MIDI export failed (non-fatal). Output:\n%s", e.stdout or "(no output)")
    except Exception as e:
        logging.warning("MIDI export failed (non-fatal): %s", e)


def main():
    parser = argparse.ArgumentParser(description="Compose suite with checkpointing and resume.")
    parser.add_argument(
        "--context-last",
        default=os.getenv("CONTEXT_LAST", "all"),
        help="How many previous model JSON bundles to include in system context per call. "
             "'all' (default), an integer N (e.g., 1), or 0 for none."
    )
    parser.add_argument(
        "--context-budget",
        type=int,
        default=int(os.getenv("CONTEXT_BUDGET_CHARS", "15000")),
        help="Max total characters from prior bundles to include in system context (default 15000)."
    )
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to an existing runs/<ts>_suite to resume.")
    parser.add_argument("--checkpoint-every", type=int, default=int(os.getenv("CHECKPOINT_EVERY", "5")),
                        help="Emit partial stitched outputs every K units (default 5).")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    # Parse --context-last
    def _parse_context_last(val: str) -> int | None:
        if val is None:
            return None
        v = str(val).strip().lower()
        if v in ("all", "", "-1"):
            return None  # None == unlimited items
        try:
            n = int(v)
            return max(0, n)
        except Exception:
            return None

    context_last = _parse_context_last(args.context_last)
    context_budget = int(args.context_budget)

    # --- Shared account + DCN session (reuse across all units) ---
    priv = os.getenv("PRIVATE_KEY")
    acct = Account.from_key(priv) if priv else Account.create("TEMPKEY for demo")
    client = DCNClient(API_BASE)
    client.ensure_auth(acct)  # single login; tokens stored in client

    # Suite dir (new or resume)
    runs_root = pathlib.Path(__file__).resolve().parent / "runs"
    runs_root.mkdir(exist_ok=True)

    if args.resume:
        suite_dir = pathlib.Path(args.resume).resolve()
        if not suite_dir.exists():
            raise SystemExit(f"--resume path not found: {suite_dir}")
        is_resume = True
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        suite_dir = runs_root / f"{ts}_suite"
        suite_dir.mkdir(parents=True, exist_ok=False)
        is_resume = False

    _setup_logging(suite_dir, args.verbose)

    # Discover templates / load from checkpoint
    if is_resume:
        ckpt = _load_json(suite_dir / "checkpoint.json", default={})
        saved = ckpt.get("templates") or []
        if saved:
            templates = [pathlib.Path(p) for p in saved]
            logging.info("Resuming with saved template order (%d files).", len(templates))
        else:
            templates = _discover_templates()
            logging.info("Resuming without saved order; rediscovered %d templates.", len(templates))

        done_labels = set(ckpt.get("done_units", []))
        ctx_bundles = list(ckpt.get("ctx_bundles", []))

        # Rehydrate finished units in order
        units_done: List[Dict[str, Any]] = []
        for p in templates:
            lbl = p.stem
            if lbl in done_labels:
                u = _load_json(suite_dir / "units" / f"{lbl}.json")
                if u:
                    units_done.append(u)

        # We will append new prompt summaries; keep the existing file as-is
        suite_pt_journal: List[Dict[str, Any]] = _load_json(suite_dir / "pt_journal.json", default=[])
        logging.info("Resume: %d/%d units already done.", len(units_done), len(templates))
    else:
        templates = _discover_templates()
        _write_json(suite_dir / "templates_order.json", [str(p) for p in templates])
        units_done: List[Dict[str, Any]] = []
        suite_pt_journal: List[Dict[str, Any]] = []
        ctx_bundles: List[str] = []
        logging.info("New run with %d templates.", len(templates))

    # Persist initial checkpoint so resume works even if we crash early
    _save_checkpoint(suite_dir, templates, units_done, ctx_bundles, args.checkpoint_every, force_partial=True)

    # Human-readable log file grows as we go
    hr_log = suite_dir / "prompts_and_summaries.txt"

    # Rolling suite_context from ctx_bundles (respect existing budget)
    suite_context = _mk_bundle_context(
        ctx_bundles, budget_chars=context_budget, max_items=context_last
    )

    logging.info("Generating units...")
    try:
        for idx, path in enumerate(tqdm(templates, desc="Generating", unit="unit", ncols=80), start=1):
            label = path.stem
            if any(u["unit_label"] == label for u in units_done):
                continue  # already done in resume

            logging.info("Unit %s (%d/%d): start", label, idx, len(templates))
            t0 = time.perf_counter()
            unit = generate_unit_from_template(
                path,
                label=label,
                fetch=False,
                suite_context=suite_context,
                acct=acct,
                dcn=client,
                session_dir=suite_dir,  # allows pt_generate to dump raw errors if needed
            )
            dt = time.perf_counter() - t0
            logging.info("Unit %s: done in %.1fs | ticks=%d", label, dt, unit["total_ticks"])

            # Merge results in-memory
            units_done.append(unit)
            suite_pt_journal.extend(unit.get("pt_journal", []))

            # Append human-readable logs
            _append_text(hr_log, f"PROMPT {unit['unit_label']}:\n{unit['rendered_user_text']}\n\n")
            _append_text(hr_log, f"SUMMARY {unit['unit_label']}:\n{unit['unit_summary_text']}\n\n")

            # Feed forward model’s minified PT JSON
            mj = unit.get("model_bundle_minjson")
            if isinstance(mj, str) and mj.strip():
                ctx_bundles.append(mj.strip())
                suite_context = _mk_bundle_context(
                    ctx_bundles, budget_chars=context_budget, max_items=context_last
                )

            # Save durable checkpoint + periodic partial outputs
            _save_checkpoint(suite_dir, templates, units_done, ctx_bundles, args.checkpoint_every)

        logging.info("All units generated. Stitching final outputs...")
        suite = _concat_units(units_done)

        _write_json(suite_dir / "composition_suite.json", suite["payload"])
        _write_json(suite_dir / "schedule.json",            suite["schedule"])
        _write_json(suite_dir / "pt_journal.json",          suite_pt_journal)

        manifest = {
            "units": [u["unit_label"] for u in units_done],
            "total_ticks": suite["total_ticks"],
            "files": {
                "payload": "composition_suite.json",
                "schedule": "schedule.json",
                "pt_journal": "pt_journal.json",
                "prompts_and_summaries": "prompts_and_summaries.txt"
            }
        }
        _write_json(suite_dir / "manifest.json", manifest)

        logging.info("Suite written:")
        logging.info("  • %s", suite_dir / "composition_suite.json")
        logging.info("  • %s", suite_dir / "schedule.json")
        logging.info("  • %s", suite_dir / "pt_journal.json")
        logging.info("  • %s", suite_dir / "manifest.json")

        _maybe_export_midi(suite_dir)

    except KeyboardInterrupt:
        logging.warning("Interrupted by user. Partial outputs and checkpoint saved in %s", suite_dir)
        _save_checkpoint(suite_dir, templates, units_done, ctx_bundles, args.checkpoint_every, force_partial=True)
    except Exception as e:
        logging.exception("Run failed. Partial outputs and checkpoint saved in %s", suite_dir)
        _save_checkpoint(suite_dir, templates, units_done, ctx_bundles, args.checkpoint_every, force_partial=True)
        raise


if __name__ == "__main__":
    main()
