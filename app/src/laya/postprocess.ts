// Port of the answer formatting in rl_agent_api.RLAgent.system_one plus
// rl_common.temp_bucket / confidence_from_probs.
import { QTYPE_NAMES, type Answer, type InternalQ, type LayaConfig } from "./types";

/** A 2-option noul and a 20-option choice need different scaling, hence the cardinality key. */
export function tempBucket(qtype: number, k: number): string {
  const size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
  return `${QTYPE_NAMES[qtype]}:${size}`;
}

export function temperatureFor(cfg: LayaConfig, qtype: number, k: number): number {
  const b = cfg.temperature_by_options[tempBucket(qtype, k)];
  return b === undefined ? cfg.temperature[qtype] : b;
}

export function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** Jev-style confidence: 1 - normalised entropy of the answer distribution. */
export function confidenceFromProbs(p: number[], k: number): number {
  if (k < 2) return 1.0;
  const q = p.slice(0, k);
  const ent = -q.reduce((a, v) => a + v * Math.log(Math.min(Math.max(v, 1e-12), 1)), 0);
  return 1 - ent / Math.log(k);
}

const r4 = (x: number) => Math.round(x * 1e4) / 1e4;

export function formatAnswer(q: InternalQ, p: number[], actProb: number): Answer {
  const k = p.length;
  const ext = { act_probability: actProb };
  const argmax = p.indexOf(Math.max(...p));
  if (q.t === "choice") {
    const keys = Object.keys((q.crit ?? {}) as Record<string, unknown>);
    return {
      type: "choice", choice: keys[argmax],
      probabilities: Object.fromEntries(keys.map((kk, i) => [kk, r4(p[i])])),
      confidence: r4(confidenceFromProbs(p, k)), rl_agent: ext,
    };
  }
  if (q.t === "score") {
    const crit = q.crit as string[];
    return {
      type: "score", score: r4(p.reduce((a, v, i) => a + i * v, 0)),
      legend: Object.fromEntries(crit.map((c, i) => [String(i), c])),
      probabilities: Object.fromEntries(p.map((v, i) => [String(i), r4(v)])),
      confidence: r4(confidenceFromProbs(p, k)), rl_agent: ext,
    };
  }
  return { type: "noul", noul: r4(p[1]), rl_agent: ext };
}
