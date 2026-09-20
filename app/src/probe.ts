// Does ORT-web actually run this graph? MatMulNBits at bits=8 and the hand-built
// int8 embedding path are the two things that could fail in the browser but not in
// onnxruntime-node. Answers that for wasm and for webgpu, against golden logits.
import * as ort from "onnxruntime-web";

type Probe = {
  case: string; input_ids: number[]; marker_pos: number[]; qtype: number;
  k: number; temperature: number; logits: number[]; probs: number[];
};

const el = document.getElementById("out")!;
const log = (s: string) => { el.textContent += s + "\n"; console.log(s); };

function softmax(z: Float64Array | number[]) {
  const m = Math.max(...z);
  const e = Array.from(z, (v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

async function tryEP(ep: "wasm" | "webgpu", probes: Probe[]) {
  const t0 = performance.now();
  // ORT-web will not fetch a graph's external .data file on its own -- it has to be
  // handed over explicitly, keyed by the exact location string stored in the graph.
  const mk = (name: string) =>
    ort.InferenceSession.create(`/models/${name}.onnx`, {
      executionProviders: [ep],
      externalData: [{ data: `/models/${name}.onnx.data`, path: `${name}.onnx.data` }],
    });
  const enc = await mk("encoder_q8");
  const head = await mk("head_q8");
  log(`[${ep}] sessions ready in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

  let maxDp = 0, maxDl = 0, agree = 0;
  for (const p of probes) {
    const L = p.input_ids.length, K = p.marker_pos.length;
    const ids = new ort.Tensor("int64", BigInt64Array.from(p.input_ids, BigInt), [1, L]);
    const att = new ort.Tensor("int64", new BigInt64Array(L).fill(1n), [1, L]);
    const t1 = performance.now();
    const { hidden } = await enc.run({ input_ids: ids, attention_mask: att });
    const out = await head.run({
      hidden,
      attention_mask: att,
      marker_pos: new ort.Tensor("int64", BigInt64Array.from(p.marker_pos, BigInt), [1, K]),
      marker_mask: new ort.Tensor("bool", new Uint8Array(K).fill(1), [1, K]),
      qtype: new ort.Tensor("int64", BigInt64Array.from([p.qtype], BigInt), [1]),
    });
    const ms = performance.now() - t1;
    const lg = Array.from(out.logits.data as Float32Array).slice(0, p.k);
    const pr = softmax(lg.map((v) => v / p.temperature));
    const dl = Math.max(...lg.map((v, i) => Math.abs(v - p.logits[i])));
    const dp = Math.max(...pr.map((v, i) => Math.abs(v - p.probs[i])));
    const ok = pr.indexOf(Math.max(...pr)) === p.probs.indexOf(Math.max(...p.probs));
    if (ok) agree++;
    maxDp = Math.max(maxDp, dp); maxDl = Math.max(maxDl, dl);
    log(`  ${p.case.padEnd(30)} L=${String(L).padStart(3)} ${ms.toFixed(0).padStart(5)}ms  d_logit ${dl.toExponential(2)}  d_p ${dp.toExponential(2)}  ${ok ? "ok" : "ARGMAX MISMATCH"}`);
  }
  log(`[${ep}] ${agree}/${probes.length} argmax | max d_logit ${maxDl.toExponential(2)} | max d_p ${maxDp.toExponential(2)}`);
}

(async () => {
  el.textContent = "";
  log(`ort ${ort.env.versions.common} | crossOriginIsolated=${self.crossOriginIsolated} | hw threads ${navigator.hardwareConcurrency}`);
  log(`webgpu available: ${"gpu" in navigator}`);
  const probes: Probe[] = await (await fetch("/probe.json")).json();
  for (const ep of ["wasm", "webgpu"] as const) {
    try { await tryEP(ep, probes); } catch (e) { log(`[${ep}] FAILED: ${e}`); }
  }
  log("done");
})();
