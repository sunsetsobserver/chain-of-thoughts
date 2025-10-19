#!/usr/bin/env node
// Usage: node tools/pt2midi.js <input_json> <output_mid>
// <input_json> MUST be the stitched payload file (composition_suite.json)

const fs  = require('fs');
const path = require('path');
const JZZ = require('jzz');
require('jzz-midi-smf')(JZZ);

// ---- Constants copied from your visualizer ----
const CHANNELS = Array.from({ length: 16 }, (_, i) => i).filter(c => c !== 9);
const GM_PROGRAMS = {
  alto_flute:   73,
  violin:       40,
  bass_clarinet:71,
  trumpet:      56,
  cello:        42,
  double_bass:  43
};
const INSTR_ORDER = [
  'alto_flute',
  'violin',
  'bass_clarinet',
  'trumpet',
  'cello',
  'double_bass'
];
const FRIENDLY_NAMES = {
  alto_flute: 'Alto Flute',
  violin: 'Violin',
  bass_clarinet: 'Bass Clarinet in Bb',
  trumpet: 'Trumpet in C',
  cello: 'Cello',
  double_bass: 'Contrabass'
};

// ---- Helpers from your web code (adapted for Node) ----
function metaFor(instrKey, INSTRUMENT_META) {
  const m = (INSTRUMENT_META || {})[instrKey] || {};
  return {
    name:    m.display_name || FRIENDLY_NAMES[instrKey] || instrKey,
    program: Number.isInteger(m.gm_program) ? m.gm_program : GM_PROGRAMS[instrKey],
    bankMSB: Number.isInteger(m.bank_msb) ? m.bank_msb
           : Number.isInteger(m.bank)     ? m.bank
           : 0,
    bankLSB: Number.isInteger(m.bank_lsb) ? m.bank_lsb : 0
  };
}

// Flatten {tracks:{instr:[{feature_path,data}...]}} → uniform array
function normalizeInput(parsed) {
  if (parsed && typeof parsed === 'object' && parsed.tracks && !Array.isArray(parsed)) {
    const out = [];
    for (const [instr, arr] of Object.entries(parsed.tracks)) {
      if (!Array.isArray(arr)) continue;
      for (const item of arr) {
        if (item && item.feature_path && !item.feature_path.startsWith(`/${instr}/`)) {
          out.push({ feature_path: `/${instr}${item.feature_path.startsWith('/') ? '' : '/'}${item.feature_path}`, data: item.data });
        } else {
          out.push(item);
        }
      }
    }
    return out;
  }
  if (Array.isArray(parsed)) return parsed;
  throw new Error('Unsupported PT JSON shape.');
}

function convertPTToMIDIEvents(ptResponse, INSTRUMENT_META) {
  // Group by "/instrument/.../scalar"
  const groups = {};
  ptResponse.forEach(({ feature_path, data }) => {
    const parts = (feature_path || '').split('/').filter(Boolean);
    if (parts.length < 2) return;
    const prefix = parts.slice(0, -1).join('/');
    const scalar = parts.at(-1);
    (groups[prefix] ||= {})[scalar] = data;
  });

  // Conductor: collect meter marks from any bucket with time+meter
  const conductorEvents = [];
  function pushTempoOnce() {
    if (!conductorEvents.some(e => e.type === 'tempo')) {
      conductorEvents.push({ type: 'tempo', time: 0, bpm: 120 });
    }
  }
  function pushTimeSig(t, num, den) {
    const last = conductorEvents.at(-1);
    if (!(last && last.type === 'timeSig' && last.time === t && last.numerator === num && last.denominator === den)) {
      conductorEvents.push({ type: 'timeSig', time: t, numerator: num, denominator: den });
    }
  }
  Object.entries(groups).forEach(([_, g]) => {
    const hasMeter = Array.isArray(g.numerator) && Array.isArray(g.denominator) && Array.isArray(g.time);
    if (hasMeter) {
      g.time.forEach((t, i) => pushTimeSig(Number(t)||0, g.numerator[i], g.denominator[i]));
    }
  });
  pushTempoOnce();
  if (!conductorEvents.some(e => e.type === 'timeSig')) {
    conductorEvents.push({ type: 'timeSig', time: 0, numerator: 4, denominator: 4 });
  }

  // Track/channel allocation
  const events = [...conductorEvents];
  const trackByInstr = new Map();
  const chanByTrack  = new Map();

  let extraStart = INSTR_ORDER.length;
  function ensureTrack(instr) {
    if (trackByInstr.has(instr)) return trackByInstr.get(instr);
    let trackIndex = INSTR_ORDER.includes(instr) ? INSTR_ORDER.indexOf(instr) : extraStart++;
    trackByInstr.set(instr, trackIndex);
    const chan = CHANNELS[trackIndex % CHANNELS.length];
    chanByTrack.set(trackIndex, chan);
    return trackIndex;
  }

  // Emit note events per instrument group with meta for program/bank later
  const noteBuckets = new Map(); // instr -> { trackIndex, channel, notes:[], meta:{} }
  Object.entries(groups).forEach(([prefix, g]) => {
    if (!g.pitch) return;
    const instr = prefix.split('/')[0];
    const trackIndex = ensureTrack(instr);
    const channel    = chanByTrack.get(trackIndex);

    let tArr = g.time;
    let dArr = g.duration;
    let vArr = g.velocity;

    if (!tArr) {
      const fallbacks = Object.entries(groups)
        .filter(([p, obj]) => p.startsWith(instr + '/') && Array.isArray(obj.time));
      const match = fallbacks.find(([, obj]) => obj.time.length === g.pitch.length);
      if (!match) return;
      tArr = match[1].time;
      dArr = dArr || match[1].duration;
      vArr = vArr || match[1].velocity;
    }

    const len = Math.min(g.pitch.length, tArr.length, (dArr?.length || Infinity), (vArr?.length || Infinity));
    const b = (noteBuckets.get(instr) || { trackIndex, channel, notes: [] });
    for (let i = 0; i < len; i++) {
      b.notes.push({
        midinote: Number(g.pitch[i])|0,
        time:     Number(tArr[i])|0,
        duration: (dArr ? Number(dArr[i])|0 : 1),
        velocity: (vArr ? Number(vArr[i])|0 : 80),
      });
    }
    noteBuckets.set(instr, b);
  });

  return { conductorEvents, noteBuckets };
}

function writeSMF(payload, outPath) {
  const INSTRUMENT_META = payload.instrument_meta || {};
  const flat = normalizeInput(payload);
  const { conductorEvents, noteBuckets } = convertPTToMIDIEvents(flat, INSTRUMENT_META);

  const PPQ = 960;
  const smf = JZZ.MIDI.SMF(1, PPQ);

  // Conductor track
  const conductor = new JZZ.MIDI.SMF.MTrk();
  smf.push(conductor);
  conductor.add(0, JZZ.MIDI.smfSeqName('PT→MIDI (Chain of Thoughts)'));

  let maxTick = 0;
  conductorEvents
    .sort((a, b) => (a.time - b.time) || (a.type === 'tempo' ? -1 : 1))
    .forEach(evt => {
      const tick = Math.round(PPQ * evt.time / 4);
      if (evt.type === 'tempo') {
        conductor.add(tick, JZZ.MIDI.smfBPM(evt.bpm));
      } else if (evt.type === 'timeSig') {
        conductor.add(tick, JZZ.MIDI.smfTimeSignature(evt.numerator, evt.denominator));
      }
      maxTick = Math.max(maxTick, tick);
    });

  // Determine present instruments, ordered like your visualizer
  const present = [...noteBuckets.keys()];
  const orderedInstrs = INSTR_ORDER.filter(i => present.includes(i))
    .concat(present.filter(i => !INSTR_ORDER.includes(i)));

  // Build per-instrument tracks
  orderedInstrs.forEach((instrKey, idx) => {
    const b = noteBuckets.get(instrKey);
    if (!b) return;
    const trk = new JZZ.MIDI.SMF.MTrk();
    smf.push(trk);

    const channel = b.channel ?? CHANNELS[idx % CHANNELS.length];
    const meta = metaFor(instrKey, INSTRUMENT_META);
    const label = `${String(idx + 1).padStart(2, '0')} ${meta.name}`;

    trk.add(0, JZZ.MIDI.smfSeqName(label));
    trk.add(0, JZZ.MIDI.smfInstrName(meta.name));
    if (meta.bankMSB || meta.bankLSB) {
      trk.add(0, JZZ.MIDI.control(channel, 0,  meta.bankMSB));
      trk.add(0, JZZ.MIDI.control(channel, 32, meta.bankLSB));
    }
    if (Number.isInteger(meta.program)) {
      trk.add(0, JZZ.MIDI.program(channel, meta.program));
    }

    b.notes.forEach(n => {
      const on  = Math.round(PPQ * n.time / 4);
      const off = Math.round(PPQ * (n.time + n.duration) / 4);
      trk.add(on,  JZZ.MIDI.noteOn (channel, n.midinote, n.velocity));
      trk.add(off, JZZ.MIDI.noteOff(channel, n.midinote, 0));
      maxTick = Math.max(maxTick, on, off);
    });

    trk.add(maxTick + 1, JZZ.MIDI.smfEndOfTrack());
  });

  conductor.add(maxTick + 1, JZZ.MIDI.smfEndOfTrack());

  // Save to disk
  const dumped = smf.dump(); // may be Uint8Array, Array<number>, or a binary string
  let buf;
  if (dumped instanceof Uint8Array) {
    buf = Buffer.from(dumped);
  } else if (Array.isArray(dumped)) {
    buf = Buffer.from(Uint8Array.from(dumped));
  } else if (typeof dumped === 'string') {
    // Older builds can return a binary string
    buf = Buffer.from(dumped, 'binary');
  } else {
    throw new Error('Unsupported SMF dump type: ' + typeof dumped);
  }
  fs.writeFileSync(outPath, buf);

}

// ---- CLI ----
const [, , inPath, outPath] = process.argv;
if (!inPath || !outPath) {
  console.error('Usage: node tools/pt2midi.js <input_json> <output_mid>');
  process.exit(2);
}
const payload = JSON.parse(fs.readFileSync(path.resolve(inPath), 'utf8'));
writeSMF(payload, path.resolve(outPath));
console.log('Wrote MIDI:', path.resolve(outPath));
