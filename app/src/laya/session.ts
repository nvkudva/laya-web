// wasm-only entry: the default entry drags in the jsep runtime, whose .wasm is 28.3MB
// and over Cloudflare Pages' 25MiB per-file cap. We only ever use the wasm EP.
import * as ort from "onnxruntime-web/wasm";
import { buildSequence, renderOptions, toInternal, type Tok } from "./sequence";
import { formatAnswer, softmax, temperatureFor } from "./postprocess";
import { loadTokenizer } from "./tokenizer";
import { QTYPES, type LayaConfig, type LayaResponse, type Questions, type State } from "./types";

export interface LoadProgress {
  file: string;
  loaded: number;
  total: number;
  cached: boolean;
}

const CACHE = "laya-weights-v1";

/** Fetch with progress, backed by the Cache API so a reload does not re-download 524MB. */
async function fetchCached(url: string, onProgress?: (p: LoadProgress) => void): Promise<Uint8Array> {
  const file = url.split("/").pop()!;
  let cache: Cache | undefined;
  try {
    cache = await caches.open(CACHE);
    const hit = await cache.match(url);
    if (hit) {
      const buf = new Uint8Array(await hit.arrayBuffer());
      onProgress?.({ file, loaded: buf.length, total: buf.length, cached: true });
      return buf;
    }
  } catch {
    // private mode / blocked storage: fall through and fetch without caching
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length") ?? 0);
  const reader = res.body!.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.({ file, loaded, total, cached: false });
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  try { await cache?.put(url, new Response(buf)); } catch { /* quota */ }
  return buf;
}

export class LayaSession {
  readonly cfg: LayaConfig;
  private tok: Tok;
  private enc: ort.InferenceSession;
  private head: ort.InferenceSession;

  private constructor(cfg: LayaConfig, tok: Tok, enc: ort.InferenceSession, head: ort.InferenceSession) {
    this.cfg = cfg;
    this.tok = tok;
    this.enc = enc;
    this.head = head;
  }

  static async load(base = "/models", onProgress?: (p: LoadProgress) => void, threads?: number): Promise<LayaSession> {
    // wasm only: ORT-web's WebGPU MatMulNBits kernel accepts 2 and 4 bits, not 8.
    // wasmPaths is not optional: left to the bundler, the production build emits the
    // asyncify and jsep variants but not the plain threaded one, and session creation
    // then hangs with no error rather than failing.
    // Production only. In dev, Vite refuses to transform a /public file that source
    // code imports, and ORT reaches these by dynamic import at runtime; left alone it
    // resolves them from node_modules, which dev serves happily.
    if (import.meta.env.PROD) ort.env.wasm.wasmPaths = "/ort/";
    ort.env.wasm.numThreads = threads ?? Math.min(navigator.hardwareConcurrency || 4, 8);
    // Main thread, no proxy. In the production bundle, threaded wasm initialises only
    // here: both a user-created worker and ORT's own env.wasm.proxy hang with no error
    // after the weights load, while the same code is fine in dev. Threads on the main
    // thread are ~7x faster than falling back to numThreads=1, so inference blocks the
    // UI for the length of one forward pass instead. See PLAN.md.
    const [cfg, tok] = await Promise.all([
      fetch(`${base}/rl_agent_config.json`).then((r) => r.json() as Promise<LayaConfig>),
      loadTokenizer(base),
    ]);
    const mk = async (name: string) => {
      const [graph, data] = [
        await fetchCached(`${base}/${name}.onnx`, onProgress),
        await fetchCached(`${base}/${name}.onnx.data`, onProgress),
      ];
      return ort.InferenceSession.create(graph, {
        executionProviders: ["wasm"],
        externalData: [{ data, path: `${name}.onnx.data` }],
      });
    };
    const enc = await mk("encoder_q8");
    const head = await mk("head_q8");
    return new LayaSession(cfg, tok, enc, head);
  }

  /** questions: {id: {type, instructions, criteria}} -- the Jev request shape. */
  async systemOne(state: State, questions: Questions): Promise<LayaResponse> {
    const out: LayaResponse = { model: "rl-agent", answers: {}, usage: { input_tokens: 0, output_tokens: 0 } };
    for (const [qid, qdef] of Object.entries(questions)) {
      const q = toInternal(qdef);
      const k = renderOptions(q).length;
      const { ids, markers } = buildSequence(this.tok, state, q, this.cfg.max_len, this.cfg.head_max_len);
      if (markers.length !== k) {
        throw new Error(`question ${JSON.stringify(qid)}: options do not fit in head_max_len=${this.cfg.head_max_len} tokens`);
      }
      const L = ids.length;
      const att = new ort.Tensor("int64", new BigInt64Array(L).fill(1n), [1, L]);
      const { hidden } = await this.enc.run({
        input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, L]),
        attention_mask: att,
      });
      const r = await this.head.run({
        hidden,
        attention_mask: att,
        marker_pos: new ort.Tensor("int64", BigInt64Array.from(markers, BigInt), [1, markers.length]),
        marker_mask: new ort.Tensor("bool", new Uint8Array(markers.length).fill(1), [1, markers.length]),
        qtype: new ort.Tensor("int64", BigInt64Array.from([QTYPES[q.t]], BigInt), [1]),
      });
      const logits = Array.from(r.logits.data as Float32Array).slice(0, k);
      const temp = temperatureFor(this.cfg, QTYPES[q.t], k);
      const p = softmax(logits.map((v) => v / temp));
      const act = softmax(Array.from(r.act_logits.data as Float32Array));
      out.answers[qid] = formatAnswer(q, p, act[0]);
      out.usage.input_tokens += L;
    }
    return out;
  }
}
