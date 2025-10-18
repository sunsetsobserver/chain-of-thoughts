"""
Prompt loading/rendering utilities:
- One global system prompt for all units (prompts/system/global.txt).
- User prompts are arbitrary template files (prompts/user/*.template.json).
"""

from string import Template
from typing import Dict, List, Optional
import json
import pathlib

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

def render_user_prompt_from_file(
    template_path: pathlib.Path,
    *,
    bar_ticks: int,
    num: int,
    den: int,
    instruments: Dict[str, Dict[str, tuple]],
    extra_vars: Optional[Dict[str, object]] = None,
) -> str:
    """
    Render ANY prompts/user/*.template.json with $-placeholders.

    Available placeholders:
      $BAR_TICKS, $NUM, $DEN, $INSTRUMENTS_JSON
    Plus anything in extra_vars (e.g., $NUM_BARS).
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    ctx = {
        "BAR_TICKS": bar_ticks,
        "NUM": num,
        "DEN": den,
        "INSTRUMENTS_JSON": json.dumps(_instruments_payload(instruments), ensure_ascii=False),
    }
    if extra_vars:
        ctx.update(extra_vars)

    return Template(_read_text(template_path)).safe_substitute(ctx)
