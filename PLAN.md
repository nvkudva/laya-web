# laya-web — Laya in the browser

## What Laya is

Not a generative LLM. A non-autoregressive **encoder + decision head** that answers
*typed questions* about a *state* in one forward pass, returning calibrated
probability distributions. No sampling, no tokens out, no hallucination surface.

Architecture (`rl_common.py::DecisionModel`):

```
input_ids ──> ModernBertModel (AutoModel, no LM head) ──> h [B,L,D]
              h += type_emb(qtype)[:,None,:]            # nn.Embedding(3, D)
              h  = 2x nn.TransformerEncoderLayer(D, nhead=D//64, ff=4D,
                                                 norm_first=True, batch_first=True)
              m  = gather(h, marker_pos)                # [B,K,D]
  logits  = scorer(m).squeeze(-1)                       # LayerNorm→Linear(D,D)→GELU→Linear(D,1)
            .masked_fill(~marker_mask, -1e4)
  feats   = [top1, top1-top2, norm_entropy, K/255]      # from softmax(logits.detach())
  act     = act_head(cat([h[:,0], feats]))              # Linear(D+4,256)→GELU→Linear(256,2)
```

Sequence layout (`build_sequence`):
`[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]`
Each option is scored at its own `[MASK]` position. Softmax over markers = the answer.

Three question types: `choice` (named options), `score` (ordinal levels, answer is
the expectation), `noul` (boolean, always rendered `[false, true]`, answer is `p[1]`).

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Checkpoint | **English root** (ModernBERT-large, 421M, 512 ctx) | best English accuracy (MASSIVE 0.783, XNLI 0.860); the only checkpoint with **fitted temperatures**, so the calibration claim is real out of the box; embeddings are only 13% of params, so int8 risk is low |
| Runtime | **onnxruntime-web**, wasm SIMD+threads | transformers.js cannot express this custom head; ORT-web's WebGPU EP does not execute `MatMulInteger`/`DynamicQuantizeLinear`, so int8 ⇒ wasm path |
| Static host | **Cloudflare Pages / Netlify** with a `_headers` file | wasm threads need `COOP: same-origin` + `COEP: credentialless`. GitHub Pages cannot set headers ⇒ single-thread wasm. `credentialless`, not `require-corp`, so the HF CDN response loads without a CORP header |
| Graphs | **two ONNX files** (encoder, head) | isolates export failures; lets each be quantized independently |
| Quantization | encoder **int8 dynamic**, head **fp32** | the scorer produces the logits the whole calibration claim rests on, and this checkpoint's `temperature = [1.637, 1.251, 1.983]` plus a populated `temperature_by_options` were fitted on fp32 logits. Quantizing the head would invalidate them |
| Tokenizer | transformers.js `AutoTokenizer` on `tokenizer.json` | HF tokenizers format loads standalone, no model needed |
| App | Vite + React + TS + **bun** | no lockfile present |
| Weight host | user's own HF repo | CORS-enabled, LFS CDN, no size cap |
| UI v1 | raw JSON request → raw JSON response | user scope call; richer UI is TODO |

## Rejected

- **typed-decisions checkpoint** — a fine-tune on one benchmark's own training split.
  Base checkpoints score 0.362/0.352 on it zero-shot vs 0.318 random. No transfer.
- **multilingual checkpoint as v1** — deferred to an on-demand second download.
  It ships uncalibrated (`temperature = [1,1,1]`, empty `temperature_by_options`),
  its `tokenizer.json` is 34MB, 61% of its params are embeddings (the part int8
  handles worst), and it is weaker on English (MASSIVE 0.657 vs 0.783).
- **typed-decisions checkpoint** — see above; a single-benchmark fine-tune.
- **Shipping both checkpoints** — ~870MB, over the stated <500MB budget.
- **transformers.js end to end** — no way to express `marker_pos` gather + custom head.
- **WebGPU fp16 variant** — only if measured wasm latency is unacceptable. Do not
  build speculatively.
- **`COEP: require-corp`** — the HF CDN does not send a CORP header, so weight
  fetches would fail. `credentialless` instead; verify in Safari before building
  the worker.
- **Single fused ONNX graph** — one export failure would be undiagnosable.

## Pipeline

1. **Golden reference.** `uv venv`, pin `transformers>=5.0` (the configs use
   `layer_types` / `rope_parameters`; 4.x silently misreads this checkpoint's split
   rope_theta — 160000 on the 10 `full_attention` layers, 10000 on the 18
   `sliding_attention` layers — with no error, producing plausible-looking garbage).
   Load `model.safetensors` strict into `DecisionModel`, run
   `RLAgent.system_one` over ~50 fixtures spanning all three qtypes, K ∈ {2,5,12},
   short and 1024-token states, non-Latin scripts. Dump token ids, marker
   positions, raw logits, post-temperature probs, act probs. Every later step is
   measured against this file.
2. **Export.** Encoder: strip the `encoder.` prefix, export `ModernBertModel` →
   `last_hidden_state`, opset ≥17, batch 1, dynamic L. Head: type_emb + 2 layers +
   gather + scorer + act_head, inputs `hidden / attention_mask / marker_pos /
   marker_mask / qtype`, dynamic L and K.
   **Trap:** `nn.TransformerEncoderLayer` takes the fused fastpath under
   eval+no_grad and will not trace — call
   `torch.backends.mha.set_fastpath_enabled(False)` before export.
3. **Quantize.** `onnxruntime.quantization.quantize_dynamic` on the encoder,
   `QInt8`, MatMul+Gather. Head untouched.
4. **Node parity gate.** `onnxruntime-node`, vs golden. Label-free, so it works
   on the 50 fixtures: argmax agreement ≥ 99%, max |Δp| ≤ 0.02, mean
   KL(p_fp32 ‖ p_int8) ≤ 1e-3. (No ECE gate — 50 fixtures over 15 bins is ~3 per
   bin, and there are no labels. ECE belongs in the later temperature-fitting
   step on a labelled set.)
   Gate fails ⇒ the suspect is the 28 encoder MatMuls, since embeddings are only
   51.6M of 421M here. Fallback ladder, in order: (a) exclude the first and last
   two encoder layers from quantization, (b) per-channel weight quantization,
   (c) fp16 head to buy back ~53MB of budget for a partly-fp16 encoder.
5. **Port `build_sequence` to TS, exactly.** The leading `" "` before each option
   text, `[:48]` per-option truncation, the `opt_budget < 16` even-shrink branch,
   `head_ids[:max(8, opt_budget)]`, the mask-token scrub on instructions/options/
   state, `truncate_left=False` for the API path. Parity test is token-id equality
   against the golden dump — not "looks right".
6. **Post-processing, exactly.** `temp_bucket(qtype, K)` → `temperature_by_options`,
   falling back to `temperature[qtype]`; softmax on `logits / T`;
   `confidence = 1 - H(p)/log(K)`; `score = Σ i·p[i]`; `noul = p[1]`;
   `act_probability = act[0]`.
7. **App.** Worker holds both sessions. Weights fetched from HF with progress,
   cached in Cache API (fall back to OPFS). Editable Jev-shaped request JSON in,
   raw response JSON out.

## Constraints

- Head must stay fp32 through every optimization pass. Anything that touches
  `scorer` or the 2 head layers invalidates the published temperatures.
- **Budget is tight: ~395MB int8 encoder + ~106MB fp32 head ≈ 501MB.** The lever
  if that is too much is an fp16 head (~53MB, total ~448MB) — but that is exactly
  the trade the fp32-head decision rejected, so measure the calibration cost
  before taking it.
- **The English checkpoint is confidently wrong on non-Latin script** — Khmer
  scores 0.000 accuracy at 0.952 confidence. Confidence gating cannot catch this.
  The UI must detect non-Latin script in the state and warn, since the model will
  not signal the failure itself.
- Expected int8 encoder size is ~395MB. If the export lands far above that,
  `Gather`/`MatMul` coverage is wrong — do not upload it.
- `tokenizer.json` is 3.5MB (50368 vocab) — cache separately from the weights.
- `max_len = 512`, `head_max_len = 192` — note these are **smaller** than the
  multilingual checkpoint's, so options and state budgets are tighter. Questions whose options do not fit must
  raise, not silently truncate the answer space — mirror the `ValueError` in
  `rl_agent_api.py`.
- Batch is fixed at 1 in the exported graphs. Multi-question requests loop.

## Revisions

- Corrected: the fp32-head rationale originally cited preserving fitted
  temperatures. The multilingual checkpoint has none — it ships
  `temperature = [1,1,1]` and an empty `temperature_by_options`. fp32 head is
  retained for a different reason (clean later fit), and temperature fitting is
  now an explicit TODO rather than an assumed property.
- Corrected: parity-gate fallback was fp16 encoder (~674MB, over budget) and is
  now per-row int8 embeddings (~197MB), which also targets the likelier culprit.
- Removed the ECE gate from step 4: unevaluable at n=50 without labels.
- Added cross-origin isolation as a hosting constraint; it rules out GitHub Pages.
- Note: `position_embedding_type: "sans_pos"` in the mmBERT config is not read by
  the ModernBert code path. If `AutoConfig` rejects it on transformers 5.x, that
  is the cause. (Not applicable to the English checkpoint, which uses `absolute`.)
- **Switched default to the English root checkpoint.** Multilingual is deferred to
  an on-demand second download. Consequences: fitted temperatures now exist, so
  the fp32-head decision recovers its original rationale; `max_len` drops 1024→512
  and `head_max_len` 256→192; `tokenizer.json` drops 34MB→3.5MB; embeddings drop
  from 61% to 13% of params, so the per-row int8 embedding fallback is no longer
  the likely fix and was replaced with a layer-exclusion ladder; total size rises
  to ~501MB; and a non-Latin-script warning becomes a UI requirement.

- **INT8 dynamic quantization (`MatMulInteger`) fails outright on this model** and is
  abandoned. Measured against golden, 26 questions:

  | Variant | argmax | max \|Δp\| | mean KL | encoder size |
  |---|---|---|---|---|
  | fp32 (reference) | 100% | 4.2e-06 | 9.0e-09 | 1582MB |
  | int8 dynamic, per-tensor | 69.2% | 0.990 | 5.6e-01 | 398MB |
  | int8 dynamic, per-channel | 76.9% | 0.995 | 4.6e-01 | 399MB |
  | int8 dynamic, **MatMul only** | 65.4% | 0.996 | 7.3e-01 | 553MB |
  | int8 dynamic, **Gather only** | 100% | 0.216 | 8.0e-03 | 1427MB |
  | weight-only NBits8, bs=32 | 100% | 0.0218 | 2.2e-04 | 595MB |
  | **weight-only NBits8, bs=64** | **100%** | **0.0158** | **1.8e-04** | **574MB** |
  | weight-only NBits8, bs=128 | 100% | 0.0194 | 1.7e-04 | 565MB |

  The ablation localises the damage: quantizing **MatMuls** alone is catastrophic
  (65.4%), quantizing **embeddings** alone is survivable (100%, but 0.216 max Δp).
  Per-channel *weight* scales barely move it, which rules out weight precision as
  the cause — the problem is **activation** quantization. ModernBERT has outlier
  activation channels that a per-tensor dynamic scale cannot represent; this is the
  same failure that motivated LLM.int8 and SmoothQuant.

  **Weight-only** quantization leaves activations in fp32 and sidesteps it. The
  weights are still 8-bit, as asked. `MatMulNBits` dequantizes inside the kernel,
  so nothing materialises a 1.6GB fp32 weight tensor in browser memory, and the op
  is supported on ORT-web's WebGPU EP — which `MatMulInteger` is not.

- **Runtime is wasm, not WebGPU.** Measured in Chromium (ORT-web 1.30, 16 threads,
  `crossOriginIsolated = true`): the wasm EP runs the q8 pair and reproduces the
  Node numbers exactly (6/6 argmax, max Δp 2.99e-02). The WebGPU EP refuses the
  graph outright — `Only 2b and 4b quantization is supported for MatMulNBits op`.
  8-bit weights therefore imply wasm. This also supersedes the earlier reasoning
  that cited `MatMulInteger`; the op is different but the conclusion is the same.

  4-bit would unlock WebGPU and cut the encoder to 271MB, but it is not viable
  here: argmax collapses to 84.6%, max Δp 0.347, mean KL 4.8e-02. Rejected.

  wasm latency, single question, M-series: 333ms at L=43, 974ms at L=195,
  2437ms at L=512. Acceptable for a playground; state length is the cost driver.

- **`act_probability` is saturated at 1.000 on all 26 fixtures.** The act/escalate
  head carries no signal on this checkpoint. Surface it in the raw JSON because the
  reference API does, but do not build UI that implies it varies.

## Revisions (2)

- Decisions "Runtime" row superseded: wasm stands, but because WebGPU's
  `MatMulNBits` is 2/4-bit only, not because of `MatMulInteger`.
- Decisions "Quantization" row superseded: weight-only NBits8 (bs=32) for MatMuls
  plus hand-built per-row-block int8 embeddings, not `quantize_dynamic`.
- Constraints "Budget" line superseded: final pair is 495.6MB (439.5 encoder +
  53.1 head + 2.9 graphs), under the 500MB target, with a bit-exact fp16-storage
  head rather than the fp32 head originally costed at 106MB.
- Gate outcome: argmax 100% and mean KL 4.5e-04 pass; max |Δp| is 0.0298 against a
  0.02 threshold. Open for the user to accept or reject.

## Revisions (3)

- Embeddings moved from per-row-block int8 to **fp16 storage**, at the user's call.
  Like the head, it is **bit-exact** (max abs error 0.000e+00) for the same reason:
  the checkpoint is bf16, and bf16's 8 mantissa bits fit inside fp16's 10.
  The `Cast` goes *after* the `Gather`, so only the rows a request touches are
  converted; casting before it would rebuild the whole 206MB fp32 table per forward.
- MatMul block size back to 64 from 32: it is both smaller and better here
  (Δp 0.0158 vs 0.0218 — the metric is a max over 26 fixtures, so it is noisy).
- **The gate now passes in full**: argmax 100%, max |Δp| 0.0158, mean KL 1.84e-04.
- Final pair **524.1MB** (470.8 encoder + 53.3 head). Over the 500MB target by 24MB,
  accepted in exchange for clearing the parity gate.

## Revisions (4)

- **The custom web worker is gone.** In the production bundle, ORT's threaded wasm
  initialises only on the main thread. Measured on the same build:

  | Where inference runs | Threads | Result |
  |---|---|---|
  | Main thread, no proxy | 8 | works — 836 ms for 3 questions |
  | Main thread, `env.wasm.proxy = true` | 8 | hangs after the weights load, no error |
  | User-created worker (module or classic) | 8 | hangs after the weights load, no error |
  | User-created worker | 1 | works — 5056 ms for 3 questions |

  All four work in `vite dev`, which is what makes this easy to ship broken. Both
  nested-worker paths fail and neither throws; the tab simply never finishes loading.
  Running on the main thread with threads is ~6x faster than the single-thread
  fallback, so inference blocks the UI for the length of one forward pass
  (~340 ms typical, ~2.4 s at the 512-token limit) rather than giving that up.
  Worth revisiting on a later ORT-web release.

- `ort.env.wasm.wasmPaths` must be set explicitly and the **jsep** runtime must be
  served alongside the plain one. The bundler emits the asyncify and jsep `.wasm`
  into `assets/` but not the plain threaded one, and the resolved ORT entry then asks
  for `ort-wasm-simd-threaded.jsep.mjs`. A missing runtime file surfaces as
  `no available backend found`, which does not name the file.
  `app/scripts/copy-ort.mjs` copies both pairs into `public/ort/` and asserts on the
  count, so a version bump that renames them fails the build instead of the page.

- Weight URLs are versioned (`/models/v1/…`). The Cache API keys on URL, so
  re-quantizing to the same path would pin every returning visitor to whatever they
  cached first, silently.

## Revisions (5)

- Import `onnxruntime-web/wasm`, not `onnxruntime-web`. The default entry pulls in the
  jsep runtime whose `.wasm` is 28.3MB — over **Cloudflare Pages' 25 MiB per-file
  cap**, so a deploy would have failed on a file nobody uses. We only ever run the
  wasm EP. `probe.html` keeps the default entry (it has to, to test WebGPU) and is
  therefore a dev-only page, excluded from the build.
- `vite.config.ts` trims `dist`: the 524MB local weight copy once `VITE_MODELS_BASE`
  is set, and the ORT `.wasm` the bundler emits into `assets/` — including a 26.9MB
  asyncify variant that is also over the cap — since `wasmPaths` always resolves to
  `/ort/`. `dist` goes from 95MB to 15MB, with no file over 25 MiB.
- Deploy build verified with local weights: 767ms for the 3-question preset.

## Revisions (6)

- Deploy configuration verified end to end against the real CDN: a 15MB site fetching
  524MB of weights cross-origin from `nvkudva/laya-web-q8` under
  `COEP: credentialless`, `crossOriginIsolated = true`, 772ms for the 3-question
  preset. **`require-corp` would have failed** — the HF CDN sends no
  `Cross-Origin-Resource-Policy` header on any of the seven files, which is exactly
  the case `credentialless` exists for.
- Tested in Chromium only. Safari and Firefox ship `credentialless` on different
  timelines; if it is unsupported the page loses `crossOriginIsolated`, wasm drops to
  one thread, and inference goes from ~770ms to ~5s rather than breaking outright.
