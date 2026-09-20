# laya-web

[Laya](https://huggingface.co/convaiinnovations/laya) running entirely in the browser,
with an 8-bit quantized copy of the English checkpoint.

Laya is not a chat model. It reads a **state**, scores the **options you enumerate**, and
returns one calibrated probability distribution per question in a single forward pass —
no sampling, no tokens out. Three question types: `noul` (true/false), `choice` (pick one
named option), `score` (expectation over ordered levels).

## Layout

```
export/        python: golden reference, ONNX export, quantization, parity verifier
golden/        fixtures + the PyTorch reference dump everything is measured against
app/           vite + react + ts playground; app/src/laya is the TS port of the model API
PLAN.md        architecture, decisions, measured results, rejected alternatives
TODO.md        task state
```

## Running it

```bash
bun run --cwd app dev
```

`/` is the playground, `/parity.html` re-runs both parity gates against the reference
dump, `/probe.html` checks which execution providers accept the graph.

The dev server sets `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy`,
which wasm threads require. `app/public/_headers` carries the same for Cloudflare Pages
and Netlify; GitHub Pages cannot set headers and will fall back to single-thread wasm.

Weights are served from `app/public/models/v1` by default. The version segment is
part of the cache key, so re-quantizing means bumping it rather than silently
serving stale weights to anyone who already cached them. Set `VITE_MODELS_BASE` to a
Hugging Face repo URL to fetch them from there instead — see `app/.env.example`.

## Rebuilding the weights

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "torch>=2.6" "transformers>=5.0" \
  safetensors numpy onnx onnxscript onnxruntime
source .venv/bin/activate
cd export
python make_fixtures.py && python dump_golden.py       # reference
python export_onnx.py                                   # fp32 encoder + head
python quantize_nbits.py --block-size 64 --out out/enc_nb8_bs64.onnx
python embed_fp16.py --src out/enc_nb8_bs64.onnx --out out/encoder_q8.onnx
python head_fp16_storage.py                             # -> out/head_q8.onnx
python verify_onnx.py --encoder out/encoder_q8.onnx --head out/head_q8.onnx --gate
```

`transformers>=5.0` is required: the encoder config uses `layer_types` and
`rope_parameters`, and 4.x misreads the split rope_theta (160000 on the 10 full-attention
layers, 10000 on the 18 sliding-attention layers) without erroring.

## What 8-bit cost

Measured over 26 fixtures against the fp32 PyTorch reference:

| | argmax agreement | max abs Δp | mean KL | size |
|---|---|---|---|---|
| fp32 reference | — | — | — | 1688 MB |
| shipped 8-bit | 100% | 0.0158 | 1.8e-04 | 524 MB |

Ordinary dynamic int8 (`quantize_dynamic`) **fails on this model** — 69% argmax agreement,
max Δp 0.99. The damage is activation quantization, not weight precision: quantizing only
the MatMuls scores 65%, quantizing only the embeddings scores 100%, and per-channel weight
scales barely move it. ModernBERT has outlier activation channels that a per-tensor dynamic
scale cannot represent. Weight-only quantization leaves activations in fp32 and avoids it.

So: MatMul weights are block-wise int8 via `MatMulNBits`, and the embeddings and decision
head are fp16 *storage* with fp32 compute. The fp16 halves are **bit-exact** — the
checkpoint is bf16, whose 8 mantissa bits fit inside fp16's 10.

This rules out WebGPU. ORT-web 1.30's WebGPU `MatMulNBits` kernel accepts 2 and 4 bits
only. 4-bit would unlock it and cut the encoder to 271 MB, but argmax collapses to 84.6%
and max Δp to 0.347. wasm it is: ~340 ms per question on short states, ~2.4 s at the
512-token limit.

Inference runs on the main thread. In a production bundle ORT's threaded wasm hangs
with no error inside a user-created worker *and* inside its own `env.wasm.proxy`
worker, while working on the main thread — and all three are fine under `vite dev`.
Single-threaded in a worker works but is ~6x slower, so the UI blocks for one forward
pass instead. `PLAN.md` has the measurements.

## Caveats

- **English only.** The checkpoint scores 0.000 accuracy at 0.952 confidence on Khmer —
  it stays confident while being wrong, so confidence gating cannot catch it. The
  playground warns when a state is mostly non-Latin script.
- **`act_probability` is saturated** at 1.000 on every fixture tested. The act/escalate
  head carries no signal on this checkpoint. It is reported because the reference API
  reports it.
- **High-cardinality choice is effectively argmax.** The published temperature for
  `choice:11+` is 0.1006, which sharpens the distribution almost to one-hot.
