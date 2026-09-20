---
license: apache-2.0
library_name: onnx
pipeline_tag: text-classification
base_model: convaiinnovations/laya
base_model_relation: quantized
tags:
  - laya
  - onnx
  - onnxruntime-web
  - quantized
  - int8
  - webassembly
  - in-browser
  - calibrated-decisions
  - classification
  - routing
  - guardrails
language:
  - en
---

# Laya, 8-bit, for the browser

**[▶ Try it — laya-web.pages.dev](https://laya-web.pages.dev)** — loads in a tab, runs
on your own machine, sends nothing anywhere.

This is [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) —
the English ModernBERT-large checkpoint — exported to ONNX and quantized to 8-bit so
it fits in a web page. **1688 MB of fp32 becomes 524 MB**, with argmax agreement
unchanged and a worst-case probability shift of 0.0158 across a 26-question parity
set.

All credit for the model, the training method and the results belongs to
**[Nandakishor M](https://github.com/NandhaKishorM)** and **Convai Innovations**.
This repository contributes only the quantization and the browser runtime.

| | |
|---|---|
| Base model | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (English root) |
| Original code | [github.com/NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) |
| This conversion | [github.com/nvkudva/laya-web](https://github.com/nvkudva/laya-web) |
| Live demo | [laya-web.pages.dev](https://laya-web.pages.dev) |
| Licence | Apache 2.0, inherited from the base model |

---

## What Laya is

Laya is **not a generative model**. It reads a *state*, scores the *options you
enumerate*, and returns one calibrated probability distribution per question in a
single forward pass. There is no sampling and no free-text output, so there is
nothing to hallucinate — the answer space is whatever you listed.

Three question types:

| Type | Answer | Use |
|---|---|---|
| `noul` | `p(true)` | Is this phishing? Should this escalate? |
| `choice` | one named option + the full distribution | Which queue? Which policy? |
| `score` | the expectation over ordered levels | How severe? How urgent? |

## Files

| File | Size | What |
|---|---|---|
| `v1/encoder_q8.onnx` + `.data` | 471 MB | ModernBERT-large encoder, 28 layers, d=1024 |
| `v1/head_q8.onnx` + `.data` | 53 MB | type embedding, 2 head layers, marker scorer, act head |
| `v1/tokenizer.json`, `v1/tokenizer_config.json` | 3.6 MB | unchanged from the base model |
| `v1/rl_agent_config.json` | — | `max_len`, `head_max_len` and the fitted temperatures |

The path is versioned on purpose. Browser caches key on URL, so re-quantizing goes
to `v2/` rather than silently serving stale weights to anyone who already has `v1/`.

## How it is quantized

**Ordinary dynamic INT8 destroys this model.** `onnxruntime.quantization.quantize_dynamic`
drops argmax agreement to **69%** with a worst-case probability shift of **0.99**. An
ablation localises the damage:

| Variant | argmax agreement | max abs Δp | mean KL |
|---|---|---|---|
| fp32 reference | — | — | — |
| dynamic int8, per-tensor | 69.2% | 0.990 | 5.6e-01 |
| dynamic int8, per-channel | 76.9% | 0.995 | 4.6e-01 |
| dynamic int8, **MatMuls only** | 65.4% | 0.996 | 7.3e-01 |
| dynamic int8, **embeddings only** | 100% | 0.216 | 8.0e-03 |
| **shipped: weight-only int8** | **100%** | **0.0158** | **1.8e-04** |

Quantizing the MatMuls alone is catastrophic while quantizing the embeddings alone
is survivable, and per-channel *weight* scales barely help — so the problem is
**activation** quantization, not weight precision. ModernBERT has outlier activation
channels that a per-tensor dynamic scale cannot represent, the same failure that
motivated LLM.int8() and SmoothQuant.

Weight-only quantization leaves activations in fp32 and avoids it entirely:

- **MatMul weights** → block-wise INT8 via `MatMulNBits` (block size 64), which
  dequantizes inside the kernel, so nothing ever materialises a 1.6 GB fp32 tensor.
- **Token embeddings and the decision head** → fp16 *storage*, fp32 compute. These
  are **bit-exact**: the original checkpoint is bf16, and bf16's 8 mantissa bits fit
  inside fp16's 10. Measured reconstruction error is exactly 0.

## Accuracy

Measured against the fp32 PyTorch model over 26 questions spanning all three types,
cardinalities 2–14, both truncation branches, non-Latin script and degenerate inputs:

| Metric | Result |
|---|---|
| Argmax agreement | **100%** (26/26) |
| Max absolute Δp | **0.0158** |
| Mean KL(fp32 ‖ int8) | **1.8e-04** |
| Tokenization | **byte-identical** token ids on all 26 |

The largest shifts land on questions the model is already uncertain about — a `noul`
sitting near p=0.5 moves most, which is where quantization error is least consequential
for a decision and most visible as a number.

## Running it

The intended consumer is [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web).
The full TypeScript port of the tokenization, sequence construction and temperature
scaling lives in [nvkudva/laya-web](https://github.com/nvkudva/laya-web) under
`app/src/laya/`.

```ts
import * as ort from "onnxruntime-web/wasm";

const BASE = "https://huggingface.co/nvkudva/laya-web-q8/resolve/main/v1";
const load = async (name: string) =>
  ort.InferenceSession.create(`${BASE}/${name}.onnx`, {
    executionProviders: ["wasm"],
    externalData: [{ data: `${BASE}/${name}.onnx.data`, path: `${name}.onnx.data` }],
  });

const encoder = await load("encoder_q8");
const head = await load("head_q8");
```

Sequence layout, which you must reproduce exactly:

```
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]
```

Each option is scored at its own `[MASK]` position; softmax over those positions,
divided by the temperature for that (question type, option count) bucket, is the
answer. `max_len` is 512 and `head_max_len` is 192.

### Deployment notes

- **WebAssembly only.** onnxruntime-web's WebGPU `MatMulNBits` kernel accepts 2-bit
  and 4-bit, not 8-bit, and rejects this graph. 4-bit would unlock WebGPU and cut the
  encoder to 271 MB, but argmax collapses to 84.6% and max Δp to 0.347 — not worth it
  for a model whose value is calibrated probabilities.
- **Cross-origin isolation is required** for wasm threads: `Cross-Origin-Opener-Policy:
  same-origin` plus `Cross-Origin-Embedder-Policy: credentialless`. Use
  `credentialless`, not `require-corp` — the Hugging Face CDN sends no
  `Cross-Origin-Resource-Policy` header, so `require-corp` blocks every file here.
  Without isolation the runtime falls back to a single thread and gets roughly 6×
  slower.
- **Import `onnxruntime-web/wasm`**, not the default entry, which pulls in a 28 MB
  jsep runtime you will not use.
- Threaded wasm initialises on the main thread but hangs silently inside a
  user-created worker and inside `env.wasm.proxy` in a production bundle.

### Performance

Chromium, Apple silicon, 8 threads, single question:

| State length | Latency |
|---|---|
| ~43 tokens | ~290 ms |
| ~195 tokens | ~920 ms |
| 512 tokens (max) | ~2.4 s |

The three-question demo preset completes in about 750 ms end to end.

## Limits

These are properties of the base model, not of the quantization, and the original
[model card](https://huggingface.co/convaiinnovations/laya) documents them fully.

- **English only.** This checkpoint scores 0.000 accuracy at 0.952 confidence on
  Khmer — it stays confident while being wrong, so confidence gating cannot catch it.
  Use [`laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)
  for anything else.
- **Near chance on typed-decisions zero-shot** (0.362 against a 0.461 majority-class
  baseline). Laya is a fast base to specialise, not a zero-shot decision engine.
- **Ordinal `score` is the weakest primitive** (SST-5 0.372).
- **High-cardinality `choice` degrades**: at `head_max_len = 192`, a 77-option
  question leaves 3–4 tokens per label. The published `choice:11+` temperature is
  0.1006, which sharpens the distribution close to one-hot.
- **`act_probability` is saturated** at 1.000 on every input tested here; the
  act/escalate head carries no signal on this checkpoint.
- **The shipped temperatures were fitted by the original author**, not refitted after
  quantization. Refitting per (question type, option count) on your own data moves
  mean ECE 0.466 → 0.081 on the base model; do that before trusting the probabilities
  in production.

## Citation

Cite the original work:

```bibtex
@misc{laya2026,
  title  = {Laya: Non-Autoregressive System 1 Decision Models},
  author = {Nandakishor M},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/convaiinnovations/laya}},
  note   = {Convai Innovations}
}
```

## Acknowledgements

[Nandakishor M](https://github.com/NandhaKishorM) and Convai Innovations built Laya,
trained it with RLCD, and released the weights and code under Apache 2.0. Read the
author's write-up
[on Dev.to](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me).

Quantization and browser runtime by [nvkudva](https://huggingface.co/nvkudva).
