# Chain of Thoughts

_write text → get music_

This repo lets you **compose multi-instrument music (as MIDI) by writing a chain of natural-language prompts describing the music as plain `.txt` files**.
You drop prompt files into `prompts/user/`. The system:

1. asks **GPT-5** (OpenAI **Responses API**) to emit a strict, machine-readable music bundle,
2. registers those instructions on the **Decentralised Creative Network (DCN)** as **Performative Transactions (PTs)**,
3. **executes** the PTs to get note arrays,
4. measures how long the generated material is,
5. **schedules & stitches** everything into one final piece: `composition_suite.json`.

> You compose by writing text; the pipeline handles schema, DCN execution, measurement, scheduling, and stitching.

---

## What are DCN and PTs? (plain English)

- **DCN (Decentralised Creative Network)** is an execution network for creative **procedures** — e.g., “generate a note stream,” “transform a parameter timeline,” “schedule events.”
- A **Performative Transaction (PT)** is one such small, typed procedure with dimensions like `time`, `duration`, `pitch`, `velocity`. Each dimension carries a list of integer ops (`add`, `subtract`, `mul`, `div`).
- When you **execute** a PT with seeds and a length `N`, DCN returns concrete arrays (onsets, durations, pitches, velocities).
- Here, GPT-5 **writes PT bundles** (one per instrument per bar). We register them on DCN and **execute** to obtain the actual data that we stitch into music.

**TL;DR:** You describe the music; GPT-5 writes the **recipe**; DCN **cooks** it. You get the MIDI file back.

---

## Requirements

- Python 3.10+
- Install deps:

  ```bash
  pip install -r requirements.txt
  ```

- **OpenAI access** to GPT-5.
- **DCN SDK** importable as `dcn` (the pipeline executes PTs via the SDK; install it per DCN docs).

---

## Configure your keys

### OpenAI key via `secrets.py`

Create **`secrets.py`** in the project root:

```python
# secrets.py
OPENAI_API_KEY = "sk-..."   # your OpenAI key
```

(Alternatively, set `OPENAI_API_KEY` in your environment.)

---

## Project layout

```
compose_suite.py      # discovers prompts, builds each unit, stitches final piece
pt_generate.py        # generates ONE unit (one prompt → 1..N bars) and saves artifacts
pt_prompts.py         # loads system prompt and reads .txt prompt files
pt_config.py          # instruments, display meta, default bar grid/meter helpers
dcn_client.py         # DCN HTTP + SDK wrapper: auth, post_feature, execute_pt

prompts/
  system/global.txt   # global system prompt (rules/specs)
  user/*.txt          # your prompts live here (filename order = suite order)

runs/                 # auto-created with rich per-run artifacts
composition_suite.json  # convenience copy of the latest stitched piece (if enabled)
```

---

## Quick start

1. **Write prompts** (plain text) in `prompts/user/`.
   Name files to control order, e.g., `001_intro.txt`, `010_chorale.txt`, `020_bridge.txt`.

   Example `prompts/user/010_chorale.txt`:

   ```
   Compose a luminous, airy chorale for six instruments that spans about ten bars.
   Keep gentle dynamics, occasional stepwise motion
   with rare leaps and corrective motion. Maintain 3/4 flow and avoid overlaps.
   Conclude with a soft cadence that feels open rather than final.
   ```

   > You can ask for “one bar” or “ten bars” or “a short section”; the system will handle the length of the generated fragment automatically.

2. **Generate**:

   ```bash
   python compose_suite.py
   ```

3. **Outputs**

   - Final stitched piece (tracks for your visualiser/renderer):

     - `runs/<timestamp>_suite/composition_suite.json`
     - (optionally also mirrored at project root if enabled in config)

   - Suite schedule (start offsets for each unit):

     - `runs/<timestamp>_suite/schedule.json`

   - Per-unit folders in `runs/` with the prompts used, raw model JSON, posted bundles, DCN receipts, executed streams, unit payload, unit schedule, and a compact **unit summary**.

---

## How continuity across prompts works

The system maintains a rolling **suite context**:

- After each unit is generated, we save a short **summary** (per-instrument note count, pitch range, last pitch/onset, average velocity, etc.) and the **prompt text** used.
- For the **next** unit, we pass that accumulated context as an extra **system** message (“CONTEXT OF THE PIECE SO FAR”), so GPT-5 composes with awareness of what has already happened.

This keeps long pieces coherent without you copy-pasting previous prompts.

---

## Bar grid & meter (defaults)

- By default, the grid is **12 ticks** per bar (i.e., 3/4) and the pipeline seeds meter accordingly for execution.
- If you need a different meter/grid, change the defaults in code (`pt_generate.py` and/or `pt_config.py`). Per-bar meters can be added later if you want that flexibility.

---

## Troubleshooting

- **`ModuleNotFoundError: dcn`** → Install the DCN SDK so `import dcn` works.
- **Model output isn’t JSON** → The system prompt enforces JSON, but if a prompt ever drifts, tighten your text (e.g., “return the bundle(s) only”).
- **Unexpected length/overlap** → The validator enforces allowed ops, strictly increasing onsets, and safe durations. If the model violates constraints, the run will raise with a helpful error near the offending feature/dimension.

---

## License

MIT
