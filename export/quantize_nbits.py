"""Weight-only block-wise int8 for the encoder MatMuls.

Plain dynamic int8 (MatMulInteger) collapses on this model: 65.4% argmax agreement,
max |dp| 0.996. The damage is activation quantization, not the weights -- ModernBERT
has outlier activation channels that a per-tensor dynamic scale cannot represent.
Weight-only quantization leaves activations in fp32 and sidesteps it entirely, and
MatMulNBits dequantizes inside the kernel so nothing materialises the fp32 weights.
"""
import argparse, os
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig

SRC = "out/encoder_fp32.onnx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--asym", action="store_true")
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--out", default="out/enc_nbits8.onnx")
    ap.add_argument("--embeddings", action="store_true", help="also quantize the embedding Gather")
    a = ap.parse_args()

    for f in (a.out, a.out + ".data"):
        if os.path.exists(f):
            os.remove(f)

    m = onnx.load(SRC)
    # accuracy_level stays None on purpose: level 4 reintroduces int8 activations,
    # which is exactly the failure this whole approach exists to avoid.
    kw = {}
    if a.embeddings:
        kw = {"op_types_to_quantize": ("MatMul", "Gather"),
              "quant_axes": (("MatMul", 0), ("Gather", 1))}
    cfg = DefaultWeightOnlyQuantConfig(block_size=a.block_size, is_symmetric=not a.asym, bits=a.bits, **kw)
    q = MatMulNBitsQuantizer(m, algo_config=cfg, **kw)
    q.process()
    onnx.save(q.model.model, a.out, save_as_external_data=True,
              location=os.path.basename(a.out) + ".data", all_tensors_to_one_file=True, size_threshold=1024)

    tot = sum(os.path.getsize(f) for f in (a.out, a.out + ".data") if os.path.exists(f))
    from collections import Counter
    c = Counter(n.op_type for n in onnx.load(a.out, load_external_data=False).graph.node)
    print("%-24s bits=%d bs=%-4d %8.1f MB  ops=%s"
          % (os.path.basename(a.out), a.bits, a.block_size, tot / 1e6,
             {k: v for k, v in c.items() if k in ("MatMulNBits", "MatMul", "Gather", "GatherBlockQuantized", "DequantizeLinear")}))


if __name__ == "__main__":
    main()
