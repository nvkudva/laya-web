"""fp16 storage for the token embedding table.

Gathers from the fp16 table and casts only the rows actually used -- casting before
the Gather would materialise the whole 206MB fp32 table on every forward pass.

Like the head, this is bit-exact: the checkpoint is bf16, whose 8 mantissa bits fit
inside fp16's 10, and the values are well inside fp16's exponent range.
"""
import argparse, os
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="out/enc_nbits8_bs64.onnx")
ap.add_argument("--out", default="out/encoder_q8.onnx")
a = ap.parse_args()
for f in (a.out, a.out + ".data"):
    if os.path.exists(f):
        os.remove(f)

m = onnx.load(a.src)
g = m.graph
inits = {i.name: i for i in g.initializer}
cand = [(n, inits[n.input[0]]) for n in g.node
        if n.op_type == "Gather" and n.input[0] in inits and len(inits[n.input[0]].dims) == 2
        and inits[n.input[0]].dims[0] > 10000]
assert len(cand) == 1, "expected exactly one embedding Gather, found %d" % len(cand)
node, tensor = cand[0]
pre = tensor.name
W = numpy_helper.to_array(tensor).astype(np.float32)
W16 = W.astype(np.float16)
err = float(np.abs(W16.astype(np.float32) - W).max())
print("embedding %s %s -> fp16, max abs error %.3e" % (pre, W.shape, err))

g.initializer.remove(tensor)
g.initializer.append(numpy_helper.from_array(W16, pre + "_f16"))
idx = list(g.node).index(node)
g.node.remove(node)
for i, n in enumerate([
    helper.make_node("Gather", [pre + "_f16", node.input[1]], [pre + "_g16"], name=pre + "_g_f16", axis=0),
    helper.make_node("Cast", [pre + "_g16"], [node.output[0]], name=pre + "_cast", to=TensorProto.FLOAT),
]):
    g.node.insert(idx + i, n)

onnx.save(m, a.out, save_as_external_data=True, location=os.path.basename(a.out) + ".data",
          all_tensors_to_one_file=True, size_threshold=1024)
mm = onnx.load(a.out, load_external_data=False)
for t in mm.graph.initializer:
    for kv in t.external_data:
        if kv.key == "location":
            kv.value = os.path.basename(a.out) + ".data"
onnx.save(mm, a.out)
print("%-24s %8.1f MB" % (os.path.basename(a.out),
                          sum(os.path.getsize(f) for f in (a.out, a.out + ".data")) / 1e6))
