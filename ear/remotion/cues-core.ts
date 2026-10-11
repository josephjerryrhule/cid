// Pure helpers for driving animation from /listen output. No dependencies.

export type Beat = { t: number; f: number; bar: number; beat: number; strength: number };
export type Hit = { t: number; f: number; v: number };
export type Moment = { t: number; f: number; type: string; strength?: number; end?: number; note?: string };
export type Note = { t: number; f: number; end: number; f1: number; pitch: number; name: string; velocity: number };
export type Tag = { label: string; group: string; t: number; f: number; end: number; f1: number; confidence: number };
export type Section = {
  id: string;
  label: string;
  start: number;
  end: number;
  f0: number;
  f1: number;
  energy: number;
  label_source: string;
};
export type Cues = {
  version: number;
  source: { file: string; hash: string; duration: number; fps: number; kind: string };
  beats: Beat[];
  hits: Record<string, Hit[]>;
  sections: Section[];
  moments: Moment[];
  notes?: Record<string, Note[]>;
  tags?: Tag[];
};
export type Curves = { fps: number; frames: number; curves: Record<string, number[]> };

type Framed = { f: number };

export function lastAtOrBefore<T extends Framed>(events: T[], frame: number): number {
  let lo = 0;
  let hi = events.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (events[mid].f <= frame) {
      ans = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return ans;
}

/** 1 on the most recent event, easing exponentially to 0. */
export function pulse(frame: number, events: Framed[], decay = 8): number {
  const i = lastAtOrBefore(events, frame);
  if (i < 0) return 0;
  return Math.exp(-(frame - events[i].f) / Math.max(decay, 1e-6));
}

export function beatPulse(
  frame: number,
  beats: Beat[],
  opts: { decay?: number; downbeatOnly?: boolean } = {},
): number {
  const list = opts.downbeatOnly ? beats.filter((b) => b.beat === 1) : beats;
  return pulse(frame, list, opts.decay ?? 8);
}

export function curveAt(curves: Curves, name: string, frame: number): number {
  const c = curves.curves[name];
  if (!c || c.length === 0) return 0;
  const x = Math.min(Math.max(frame, 0), c.length - 1);
  const i = Math.floor(x);
  const j = Math.min(i + 1, c.length - 1);
  return c[i] + (c[j] - c[i]) * (x - i);
}

export function momentAt(frame: number, moments: Moment[], type: string, tolerance = 0): Moment | null {
  for (const m of moments) {
    if (m.type === type && Math.abs(frame - m.f) <= tolerance) return m;
  }
  return null;
}

export function sectionAt(frame: number, sections: Section[]): Section | null {
  return sections.find((s) => s.f0 <= frame && frame < s.f1) ?? null;
}

/** The note sounding at this frame (binary search on start frame, then range check). */
export function noteAt(frame: number, notes: Note[]): Note | null {
  const i = lastAtOrBefore(notes, frame);
  return i >= 0 && frame < notes[i].f1 ? notes[i] : null;
}

export function tagActive(frame: number, tags: Tag[], label: string): Tag | null {
  return tags.find((s) => s.label === label && s.f <= frame && frame < s.f1) ?? null;
}
