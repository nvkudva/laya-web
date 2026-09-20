"""INT8-quantize the encoder only. The head stays fp32 -- this checkpoint's fitted
temperatures (1.637 / 1.251 / 1.983 plus temperature_by_options) were computed on
fp32 logits, and quantizing the scorer would invalidate them.
"""
import argparse, os, shutil
from onnxruntime.quantization import quantize_dynamic, QuantType

SRC = "out/encoder_fp32.onnx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/encoder_int8.onnx")
    ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--exclude-edge-layers", type=int, default=0,
                    help="exclude the first and last N encoder layers from quantization")
    ap.add_argument("--ops", default="", help="comma list, e.g. MatMul or Gather")
    a = ap.parse_args()

    nodes_to_exclude = []
    if a.exclude_edge_layers:
        import onnx
        m = onnx.load(SRC, load_external_data=False)
        pref = tuple("layers.%d." % i for i in
                     list(range(a.exclude_edge_layers)) + list(range(28 - a.exclude_edge_layers, 28)))
        nodes_to_exclude = [n.name for n in m.graph.node
                            if n.op_type == "MatMul" and any(p in n.name for p in pref)]
        print("excluding %d MatMul nodes in edge layers" % len(nodes_to_exclude))

    for f in (a.out, a.out + ".data"):
        if os.path.exists(f):
            os.remove(f)

    ops = [x for x in a.ops.split(",") if x] or None
    quantize_dynamic(SRC, a.out, weight_type=QuantType.QInt8, per_channel=a.per_channel,
                     nodes_to_exclude=nodes_to_exclude, op_types_to_quantize=ops,
                     extra_options={"EnableSubgraph": False, "MatMulConstBOnly": True})

    tot = sum(os.path.getsize(f) for f in (a.out, a.out + ".data") if os.path.exists(f))
    src = sum(os.path.getsize(f) for f in (SRC, SRC + ".data") if os.path.exists(f))
    print("%-28s %8.1f MB  (fp32 %.1f MB, %.2fx)" % (os.path.basename(a.out), tot / 1e6, src / 1e6, src / tot))

    import onnx
    m = onnx.load(a.out, load_external_data=False)
    from collections import Counter
    c = Counter(n.op_type for n in m.graph.node)
    print("quantized ops:", {k: v for k, v in c.items() if k in
                             ("MatMulInteger", "DynamicQuantizeLinear", "Gather", "MatMul", "DequantizeLinear")})


if __name__ == "__main__":
    main()
