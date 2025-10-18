# pt_prompts.py
"""
Prompt loading/rendering utilities (TXT-friendly):

- One global system prompt for all units: prompts/system/global.txt
- User prompts can be plain .txt (or templates with $-placeholders if you like).
- Exposes both render_user_prompt_file(...) and render_user_prompt_from_file(...)
  (the latter is a back-compat alias).
"""

from string import Template
from typing import Dict, List
import json
import pathlib
import re

# Anchors to file locations relative to this file's folder.
BASE_DIR = pathlib.Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"


def _read_text(path: pathlib.Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_system_prompt(fallback: str = "") -> str:
    """
    Load a single global system prompt used for all units.
    File path: prompts/system/global.txt
    """
    p = PROMPTS_DIR / "system" / "global.txt"
    return _read_text(p) if p.exists() else fallback


def _instruments_payload(instruments: Dict[str, Dict[str, tuple]]) -> List[Dict]:
    return [
        {"id": k, "range": list(v["range"]), "tessitura": list(v["tess"])}
        for k, v in instruments.items()
    ]


def _render_text_core(
    template_path: pathlib.Path,
    *,
    bar_ticks: int,
    num: int,
    den: int,
    instruments: Dict[str, Dict[str, tuple]],
) -> str:
    """
    Render ANY prompts/user/* file. If it contains $-placeholders, we substitute:
      $BAR_TICKS, $NUM, $DEN, $INSTRUMENTS_JSON
    If not, the text passes through unchanged.
    """
    text = _read_text(template_path)
    ctx = {
        "BAR_TICKS": bar_ticks,
        "NUM": num,
        "DEN": den,
        "INSTRUMENTS_JSON": json.dumps(_instruments_payload(instruments), ensure_ascii=False),
    }
    # Always safe_substitute: if no placeholders, text is returned unchanged.
    return Template(text).safe_substitute(ctx)


def render_user_prompt_file(
    template_path: pathlib.Path,
    *,
    bar_ticks: int,
    num: int,
    den: int,
    instruments: Dict[str, Dict[str, tuple]],
) -> str:
    """Primary entry point used by pt_generate.py."""
    return _render_text_core(
        template_path,
        bar_ticks=bar_ticks,
        num=num,
        den=den,
        instruments=instruments,
    )


# --- Back-compat alias (some older scripts may import this name) ---
def render_user_prompt_from_file(
    template_path: pathlib.Path,
    *,
    bar_ticks: int,
    num: int,
    den: int,
    instruments: Dict[str, Dict[str, tuple]],
    extra_vars=None,  # ignored in TXT mode
) -> str:
    return _render_text_core(
        template_path,
        bar_ticks=bar_ticks,
        num=num,
        den=den,
        instruments=instruments,
    )

def parse_prompt_directives(path: pathlib.Path) -> dict:
    """
    Optional inline directives in the prompt text:
      METER: <NUM>/<DEN>        (e.g., "METER: 5/8")
      BAR_TICKS: <int>          (e.g., "BAR_TICKS: 10")
    Returns a dict like {"num": 5, "den": 8, "bar_ticks": 10} for whatever was found.
    """
    text = _read_text(path)
    out = {}
    m = re.search(r"(?im)^\s*meter\s*:\s*(\d+)\s*/\s*(\d+)\s*$", text)
    if m:
        out["num"] = int(m.group(1))
        out["den"] = int(m.group(2))
    t = re.search(r"(?im)^\s*bar_ticks\s*:\s*(\d+)\s*$", text)
    if t:
        out["bar_ticks"] = int(t.group(1))
    return out
