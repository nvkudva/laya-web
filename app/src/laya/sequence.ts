// Port of rl_common.render_options / serialize_state / build_sequence.
// Token-for-token equality with the Python original is the contract here; every
// slice bound and stray space below is load-bearing and verified against golden.json.
import type { InternalQ, QuestionDef, State } from "./types";

export interface Tok {
  encode(text: string, opts: { add_special_tokens: boolean }): number[];
  maskToken: string;
  maskTokenId: number;
  clsTokenId: number;
  sepTokenId: number;
  padTokenId: number;
}

/** Python's json.dumps: separators default to ", " and ": ", unlike JSON.stringify. */
export function pyJsonDumps(v: unknown, ensureAscii: boolean): string {
  const esc = (s: string) => {
    let out = '"';
    for (const ch of s) {
      const c = ch.codePointAt(0)!;
      if (ch === '"') out += '\\"';
      else if (ch === "\\") out += "\\\\";
      else if (ch === "\n") out += "\\n";
      else if (ch === "\r") out += "\\r";
      else if (ch === "\t") out += "\\t";
      else if (c < 0x20) out += "\\u" + c.toString(16).padStart(4, "0");
      else if (ensureAscii && c > 0x7f) {
        if (c > 0xffff) {
          const u = c - 0x10000;
          out += "\\u" + (0xd800 + (u >> 10)).toString(16).padStart(4, "0");
          out += "\\u" + (0xdc00 + (u & 0x3ff)).toString(16).padStart(4, "0");
        } else out += "\\u" + c.toString(16).padStart(4, "0");
      } else out += ch;
    }
    return out + '"';
  };
  const go = (x: unknown): string => {
    if (x === null || x === undefined) return "null";
    if (typeof x === "string") return esc(x);
    if (typeof x === "boolean") return x ? "true" : "false";
    if (typeof x === "number") return Number.isInteger(x) ? String(x) : String(x);
    if (Array.isArray(x)) return "[" + x.map(go).join(", ") + "]";
    return "{" + Object.entries(x as object).map(([k, v]) => esc(k) + ": " + go(v)).join(", ") + "}";
  };
  return go(v);
}

export function serializeState(state: State): string {
  return typeof state === "string" ? state : pyJsonDumps(state, false);
}

/** Jev request -> internal question. Mirrors RLAgent._to_internal. */
export function toInternal(q: QuestionDef): InternalQ {
  let crit = q.criteria ?? null;
  if (q.type === "choice" && Array.isArray(crit)) {
    const d: Record<string, string | null> = {};
    for (const c of crit) d[c] = null;
    crit = d;
  }
  // note: this json.dumps has ensure_ascii at its default of true, unlike serialize_state
  const ins = typeof q.instructions === "string" ? q.instructions : pyJsonDumps(q.instructions, true);
  return { t: q.type, ins, crit };
}

/** Option texts in label-index order. Noul is always [false, true] so p[1] == noul. */
export function renderOptions(q: InternalQ): string[] {
  if (q.t === "choice") {
    const crit = (q.crit ?? {}) as Record<string, string | null>;
    // `if not v` in Python: null, undefined and "" all fall back to the bare key
    return Object.entries(crit).map(([k, v]) => (!v ? k : `${k}: ${v}`));
  }
  if (q.t === "score") {
    return (q.crit as string[]).map((c, i) => `level ${i}: ${c}`);
  }
  const crit = (q.crit ?? {}) as Record<string, string | null>;
  return [
    "false: " + (crit["false"] || "no, the statement does not hold"),
    "true: " + (crit["true"] || "yes, the statement holds"),
  ];
}

export interface Sequence {
  ids: number[];
  markers: number[];
}

/**
 * [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
 * Returns the token ids and the position of each option's [MASK] marker.
 */
export function buildSequence(
  tok: Tok, state: State, q: InternalQ, maxLen: number, headMaxLen: number,
  truncateLeft = false,
): Sequence {
  const scrub = (s: string) => s.split(tok.maskToken).join(" ");
  const opts = renderOptions(q);
  const ins = scrub(String(q.ins));
  let headIds = tok.encode(`${q.t} question: ${ins}`, { add_special_tokens: false });

  let optIds = opts.map((o) => [
    tok.maskTokenId,
    ...tok.encode(" " + scrub(o), { add_special_tokens: false }).slice(0, 48),
  ]);
  let optBudget = headMaxLen - optIds.reduce((a, o) => a + o.length, 0);
  if (optBudget < 16) {
    // too many / too long options: shrink every option text evenly
    const per = Math.max(4, Math.floor((headMaxLen - 16) / Math.max(1, optIds.length)));
    optIds = optIds.map((o) => o.slice(0, per));
    optBudget = headMaxLen - optIds.reduce((a, o) => a + o.length, 0);
  }
  headIds = headIds.slice(0, Math.max(8, optBudget));

  const ids = [tok.clsTokenId, ...headIds, tok.sepTokenId];
  const markers: number[] = [];
  for (const o of optIds) {
    markers.push(ids.length);
    ids.push(...o);
  }
  ids.push(tok.sepTokenId);

  const room = Math.max(0, maxLen - ids.length - 1);
  const stAll = tok.encode(scrub(serializeState(state)), { add_special_tokens: false });
  // Python's st[-0:] is the whole list, not the empty one -- reproduce that, not slice(-0)
  const st = truncateLeft ? (room === 0 ? stAll : stAll.slice(-room)) : stAll.slice(0, room);
  const all = [...ids, ...st, tok.sepTokenId];
  return { ids: all.slice(0, maxLen), markers: markers.filter((m) => m < maxLen) };
}
