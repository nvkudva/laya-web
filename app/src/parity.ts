// Two gates in one page:
//   1. tokenizer + build_sequence produce byte-identical token ids to the PyTorch
//      reference. Anything less and every number after it is meaningless.
//   2. end-to-end systemOne output matches the reference response.
import { buildSequence, renderOptions, toInternal } from "./laya/sequence";
import { loadTokenizer } from "./laya/tokenizer";
import { LayaSession } from "./laya/session";
import type { LayaConfig, Questions, State } from "./laya/types";

interface Expect { input_ids: number[]; marker_pos: number[]; rendered_options: string[]; probs: number[]; k: number }
interface Case { id: string; state: State; questions: Questions; expect: Record<string, Expect>; response: any }

const el = document.getElementById("out")!;
const log = (s = "") => { el.textContent += s + "\n"; };
const eq = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i]);

function firstDiff(a: number[], b: number[]) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i;
  return n;
}

(async () => {
  el.textContent = "";
  const data: { cfg: LayaConfig; cases: Case[] } = await (await fetch("/parity.json")).json();
  const tok = await loadTokenizer("/models");

  log("=== gate 1: token ids and marker positions ===");
  let nq = 0, idsOk = 0, mkOk = 0, optOk = 0;
  const failures: string[] = [];
  for (const c of data.cases) {
    for (const [qid, qdef] of Object.entries(c.questions)) {
      nq++;
      const q = toInternal(qdef);
      const opts = renderOptions(q);
      const e = c.expect[qid];
      const { ids, markers } = buildSequence(tok, c.state, q, data.cfg.max_len, data.cfg.head_max_len);
      const o = JSON.stringify(opts) === JSON.stringify(e.rendered_options);
      const i = eq(ids, e.input_ids);
      const m = eq(markers, e.marker_pos);
      if (o) optOk++; if (i) idsOk++; if (m) mkOk++;
      if (!(o && i && m)) {
        const d = firstDiff(ids, e.input_ids);
        failures.push(`  ${c.id}/${qid}  opts:${o ? "ok" : "DIFF"} ids:${i ? "ok" : `DIFF@${d} len ${ids.length} vs ${e.input_ids.length}`} markers:${m ? "ok" : `${markers} vs ${e.marker_pos}`}`);
        if (!o) failures.push(`    got  ${JSON.stringify(opts)}\n    want ${JSON.stringify(e.rendered_options)}`);
        if (!i) failures.push(`    got  [...${ids.slice(Math.max(0, d - 2), d + 4)}...]\n    want [...${e.input_ids.slice(Math.max(0, d - 2), d + 4)}...]`);
      }
    }
  }
  log(`options ${optOk}/${nq}   ids ${idsOk}/${nq}   markers ${mkOk}/${nq}`);
  failures.forEach((f) => log(f));
  if (idsOk !== nq || mkOk !== nq || optOk !== nq) { log("\nGATE 1 FAILED - stopping"); return; }
  log("GATE 1 PASS\n");

  log("=== gate 2: end-to-end vs reference response ===");
  const t0 = performance.now();
  const laya = await LayaSession.load("/models", (p) => {
    if (p.loaded === p.total) el.textContent = el.textContent.replace(/\n?loading.*$/, "") + `\nloading ${p.file} done${p.cached ? " (cached)" : ""}`;
  });
  log(`\nsession ready in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  let maxDp = 0, argmaxOk = 0, total = 0, ms = 0;
  for (const c of data.cases) {
    const t = performance.now();
    const res = await laya.systemOne(c.state, c.questions);
    ms += performance.now() - t;
    for (const [qid, e] of Object.entries(c.expect)) {
      total++;
      const a = res.answers[qid] as any;
      const got: number[] = Object.values(a.probabilities ?? { 0: 1 - a.noul, 1: a.noul });
      const dp = Math.max(...got.map((v, i) => Math.abs(v - e.probs[i])));
      maxDp = Math.max(maxDp, dp);
      if (got.indexOf(Math.max(...got)) === e.probs.indexOf(Math.max(...e.probs))) argmaxOk++;
      if (dp > 0.02) log(`  ${c.id}/${qid}  d_p ${dp.toFixed(4)}`);
    }
  }
  log(`\n${total} questions | argmax ${((argmaxOk / total) * 100).toFixed(1)}% | max d_p ${maxDp.toExponential(2)} | ${(ms / data.cases.length).toFixed(0)}ms per request`);
  log(argmaxOk === total && maxDp <= 0.02 ? "GATE 2 PASS" : "GATE 2 FAILED");
})().catch((e) => log("ERROR: " + (e?.stack ?? e)));
