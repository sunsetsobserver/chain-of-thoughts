"""
Central config/constants for PT generation.
Every line is commented to keep future you happy.
"""

from typing import Tuple, Dict

# Base URL of your DCN API.
API_BASE: str = "https://api.decentralised.art"

# Fixed instrument order for routing/visualiser track assembly.
ORDERED_INSTRS = ["alto_flute", "violin", "bass_clarinet", "trumpet", "cello", "double_bass"]

# Hard ranges and tessituras (MIDI numbers) for each instrument.
INSTRUMENTS: Dict[str, Dict[str, tuple]] = {
    "alto_flute":    {"range": (53, 81),  "tess": (55, 79)},
    "violin":        {"range": (55, 88),  "tess": (60, 84)},
    "bass_clarinet": {"range": (43, 74),  "tess": (48, 70)},
    "trumpet":       {"range": (60, 82),  "tess": (60, 78)},
    "cello":         {"range": (48, 74),  "tess": (50, 69)},
    "double_bass":   {"range": (31, 55),  "tess": (33, 50)},
}

# Visualiser metadata per instrument (GM program numbers etc.).
INSTRUMENT_META = {
    "alto_flute":    {"display_name": "Alto Flute",         "gm_program": 73, "bank": 0},
    "violin":        {"display_name": "Violin",             "gm_program": 40, "bank": 0},
    "bass_clarinet": {"display_name": "Bass Clarinet in Bb","gm_program": 71, "bank": 0},
    "trumpet":       {"display_name": "Trumpet in C",       "gm_program": 56, "bank": 0},
    "cello":         {"display_name": "Cello",              "gm_program": 42, "bank": 0},
    "double_bass":   {"display_name": "Contrabass",         "gm_program": 43, "bank": 0},
}

# Optional: ticks per bar per bar-index (expand as you add bars).
# 1 tick = 1/16 note; 12 means 3/4 with 16th grid.
BAR_TICKS_BY_BAR = {
    1: 12,   # 3/4
    2: 8,    # 2/4  (example)
    3: 16,   # 4/4  (example)
}

def meter_from_ticks(ticks: int) -> Tuple[int, int]:
    """
    Map ticks to (numerator, denominator).
    Extend this dict if you use other metres on a 16th grid.
    """
    return {12:(3,4), 8:(2,4), 16:(4,4), 4:(1,4)}.get(int(ticks), (4,4))
