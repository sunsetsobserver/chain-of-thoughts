# Chain of Thoughts

_write text → get music_

This repo lets you **compose multi-instrument music (as MIDI) by writing a chain of plain-text prompts**.
Drop `.txt` prompts into `prompts/user/`. The system:

1. asks **GPT-5** (OpenAI **Responses API**) to emit a strict, machine-readable music bundle,
2. registers those instructions on the **Decentralised Creative Network (DCN)** as **Performative Transactions (PTs)**,
3. **executes** the PTs to get note arrays,
4. **stitches** all units into one piece (`composition_suite.json`), and
5. (optionally) **exports a .mid** using a small Node tool.

> You compose in text; the pipeline handles schema, DCN execution, validation, scheduling, and stitching.

---

## What are DCN and PTs? (plain English)

- **DCN (Decentralised Creative Network)** executes creative **procedures** (PTs) like “generate a note stream.”
- A **Performative Transaction (PT)** has dimensions (`time`, `duration`, `pitch`, `velocity`, `numerator`, `denominator`), each a list of integer ops (`add`, `subtract`, `mul`, `div`).
- When you **execute** a PT with seeds and length `N`, DCN returns concrete arrays.
- Here, GPT-5 **writes PT bundles** (one per instrument per bar). We post them to DCN and execute to obtain the actual notes.

**TL;DR:** You describe the music → GPT-5 writes the **recipe** → DCN **cooks** it → you get JSON (and optionally MIDI).

---

## Requirements

- **Python 3.10+**
- Python deps:

  ```bash
  pip install -r requirements.txt
  ```

- **OpenAI** access to GPT-5
- **DCN SDK** importable as `dcn` (install per DCN docs)
- _(Optional, for MIDI export)_ **Node 18+** with `jzz` and `jzz-midi-smf` (installed via `npm install` in this repo)

---

## Configure your keys

Create **`secrets.py`** in the project root:

```python
# secrets.py
OPENAI_API_KEY = "sk-..."   # your OpenAI key
```

(Alternatively, set `OPENAI_API_KEY` in your environment.)

For DCN auth, the pipeline will use `PRIVATE_KEY` from the environment if present; otherwise it creates a temporary account for the session.

---

## Project layout

```
compose_suite.py      # discovers prompts, builds each unit, stitches final piece
pt_generate.py        # generates ONE unit (one prompt → 1..N bars), returns data to compose_suite
pt_prompts.py         # loads system prompt and reads .txt prompt files; parses METER from text
pt_config.py          # instruments meta & helpers (ranges, display info)
dcn_client.py         # DCN HTTP + SDK wrapper: auth, post_feature, execute_pt
tools/pt2midi.js      # (Node) PT-JSON → .mid writer using jzz + jzz-midi-smf

prompts/
  system/global.txt   # global system prompt (composer persona + hard rules)
  user/*.txt          # your prompts (filename order = suite order)

runs/                 # auto-created per full suite run with all artifacts
```

---

## Writing prompts (how to control meter)

**Meter is specified in your prompt text** via a simple directive at the top:

```
METER: 3/4
```

Supported mappings on a 1/16 grid (ticks per bar):

- `3/4` → **12** ticks
- `4/4` → **16** ticks
- `2/4` → **8** ticks
- `1/4` → **4** ticks

If you omit `METER:`, the unit defaults to **3/4 (12 ticks)**.
Advanced: you can also force `BAR_TICKS: <int>`. The system appends exact hard MIDI ranges and a meter reminder to each user prompt automatically.

**Ordering:** files in `prompts/user/` are processed lexicographically; use numeric prefixes (e.g., `001_intro.txt`, `010_clouds.txt`, …).

**Example prompt**

```text
METER: 3/4

TITLE
Airy chorale — six parts, soft dynamics.

INSTRUMENTS (EXACT)
[alto_flute, violin, bass_clarinet, trumpet, cello, double_bass].

CONSTRAINTS
Monophony per instrument; time uses only add {1,2,3,4}; durations fit gaps; pitch add/sub only; no overlaps.

GOAL
A luminous, stepwise texture with occasional small leaps and corrective motion. Close but non-triadic vertical colors.
```

---

## Quick start

1. **Write prompts** in `prompts/user/` (include `METER: ...` in the text when you need a meter change).
2. **Generate the suite:**

```bash
python compose_suite.py
```

**Outputs (per run) in `runs/<timestamp>_suite/`:**

- `composition_suite.json` — the stitched, multi-track PT output (for visualisers/MIDI export)
- `schedule.json` — unit start offsets & meters
- `pt_journal.json` — compact log of posted/executed PTs
- `prompts_and_summaries.txt` — the rendered prompts and computed summaries
- `manifest.json` — filenames & totals

_(Optionally mirrored elsewhere by your own scripts.)_

---

## MIDI export

This repo ships with a small Node tool that converts the stitched PT JSON to a Standard MIDI File.

### One-time setup

```bash
npm install
```

### New runs (automatic)

When you run:

```bash
python compose_suite.py
```

the pipeline writes:

- `runs/<ts>_suite/composition_suite.json`
- `runs/<ts>_suite/composition_suite.mid` ← **MIDI (auto)**
- plus the usual logs and summaries

> If Node isn’t installed or dependencies are missing, the run still completes; only the MIDI step is skipped with a warning.

### Past runs (retroactive)

You can create a `.mid` for any earlier suite folder:

```bash
node tools/pt2midi.js runs/<ts>_suite/composition_suite.json runs/<ts>_suite/composition_suite.mid
```

**Latest run quick command (bash):**

```bash
latest="$(ls -dt runs/*_suite | head -1)"
node tools/pt2midi.js "$latest/composition_suite.json" "$latest/composition_suite.mid"
```

**Batch all runs (bash):**

```bash
for d in runs/*_suite; do
  [ -f "$d/composition_suite.json" ] && \
  node tools/pt2midi.js "$d/composition_suite.json" "$d/composition_suite.mid"
done
```

> The converter uses the per-instrument GM programs/banks from `instrument_meta` (if present). It also embeds time-signature marks from the `numerator`/`denominator` streams.

## How continuity across prompts works

After each unit, we capture the model’s **raw PT JSON** (minified) and feed **the entire accumulated history** of those bundles into the next call as a special _reference_ system message.

- The model sees exactly what it previously emitted (bars/sections, features, run_plan, seeds, etc.).
- We instruct it **not to echo** those prior objects; it must return **only one JSON object** for the current unit.
- Literal reuse becomes possible: when you ask to “reuse the Background Loop from the reference bundle that has N=… (transpose +2, rename),” the model can copy the prior structure.
- We no longer pass earlier user prompts or summaries into the model’s context. Those remain in `runs/<ts>_suite/prompts_and_summaries.txt` for human reading.

---

## Validation & guardrails (what the generator enforces)

- **Allowed ops:** `add`, `subtract`, `mul`, `div` (exact spelling).
- **Time:** strictly increasing, `add {1,2,3,4}` only (no zeros, no chords).
- **Duration:** live values must be in `{1,2,3,4}`; automatically **capped to next onset and bar end** for safety.
- **Pitch:** `add`/`subtract` only; kept inside **hard MIDI ranges** (per instrument).
- **Meter:** seeds set from your `METER:` directive; meter dims use constant `add 0`.
- **Monophony:** enforced via time/duration rules and capping.

If a bundle violates constraints, the run raises with a clear error pointing to the offending feature/dimension.

---

## Troubleshooting

- `ModuleNotFoundError: dcn` → Install the DCN SDK so `import dcn` works.
- Model output isn’t valid JSON → Tighten the prompt (“return the bundle(s) only; one JSON object; no prose”).
- Notes overlap or spill → The runner caps durations to the next onset/bar end, but keep your durations ≤ smallest time step to remain musical.
- MIDI export fails → Ensure Node 18+ and `npm install` were executed; check the input path to `composition_suite.json`.

---

## License

MIT
