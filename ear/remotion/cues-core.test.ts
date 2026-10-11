import assert from "node:assert/strict";
import { test } from "node:test";
import {
  beatPulse,
  curveAt,
  lastAtOrBefore,
  momentAt,
  pulse,
  sectionAt,
  type Beat,
  type Curves,
  type Moment,
  type Section,
  noteAt,
  tagActive,
  type Note,
  type Tag,
} from "./cues-core.ts";

const beats: Beat[] = [
  { t: 0.5, f: 30, bar: 1, beat: 1, strength: 0.9 },
  { t: 1.0, f: 60, bar: 1, beat: 2, strength: 0.6 },
  { t: 1.5, f: 90, bar: 1, beat: 3, strength: 0.7 },
  { t: 2.0, f: 120, bar: 1, beat: 4, strength: 0.6 },
  { t: 2.5, f: 150, bar: 2, beat: 1, strength: 0.9 },
];

test("lastAtOrBefore uses binary search semantics", () => {
  assert.equal(lastAtOrBefore(beats, 0), -1);
  assert.equal(lastAtOrBefore(beats, 30), 0);
  assert.equal(lastAtOrBefore(beats, 89), 1);
  assert.equal(lastAtOrBefore(beats, 1000), 4);
});

test("pulse is 1 on the event and decays", () => {
  assert.equal(pulse(30, beats), 1);
  assert.ok(pulse(38, beats, 8) < 0.4);
  assert.equal(pulse(10, beats), 0);
});

test("beatPulse downbeatOnly ignores other beats", () => {
  assert.equal(beatPulse(60, beats), 1);
  assert.ok(beatPulse(60, beats, { downbeatOnly: true }) < 0.1);
  assert.equal(beatPulse(150, beats, { downbeatOnly: true }), 1);
});

test("curveAt interpolates, clamps and tolerates missing curves", () => {
  const curves: Curves = { fps: 60, frames: 3, curves: { "bands.low": [0, 1, 0.5] } };
  assert.equal(curveAt(curves, "bands.low", 0.5), 0.5);
  assert.equal(curveAt(curves, "bands.low", -10), 0);
  assert.equal(curveAt(curves, "bands.low", 99), 0.5);
  assert.equal(curveAt(curves, "nope", 1), 0);
});

test("momentAt and sectionAt", () => {
  const moments: Moment[] = [{ t: 30.97, f: 1858, type: "drop", strength: 0.97 }];
  assert.equal(momentAt(1856, moments, "drop", 2)?.f, 1858);
  assert.equal(momentAt(1850, moments, "drop", 2), null);
  const sections: Section[] = [
    { id: "s1", label: "intro", start: 0, end: 15.48, f0: 0, f1: 929, energy: 0.2, label_source: "auto" },
    { id: "s2", label: "build", start: 15.48, end: 30.97, f0: 929, f1: 1858, energy: 0.5, label_source: "gemini" },
  ];
  assert.equal(sectionAt(929, sections)?.id, "s2");
  assert.equal(sectionAt(5000, sections), null);
});

test("noteAt returns the sounding note", () => {
  const notes: Note[] = [
    { t: 1, f: 60, end: 1.4, f1: 84, pitch: 69, name: "A4", velocity: 0.6 },
    { t: 2, f: 120, end: 2.4, f1: 144, pitch: 72, name: "C5", velocity: 0.7 },
  ];
  assert.equal(noteAt(70, notes)?.name, "A4");
  assert.equal(noteAt(100, notes), null);
  assert.equal(noteAt(120, notes)?.name, "C5");
});

test("tagActive finds a span for a label", () => {
  const tags: Tag[] = [{ label: "Guitar", group: "instrument", t: 0, f: 0, end: 10, f1: 600, confidence: 0.8 }];
  assert.equal(tagActive(300, tags, "Guitar")?.label, "Guitar");
  assert.equal(tagActive(700, tags, "Guitar"), null);
  assert.equal(tagActive(300, tags, "Speech"), null);
});
