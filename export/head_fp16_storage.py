"""Store the head's weights as fp16, cast back to fp32 in-graph.

Compute stays fp32 -- this is a storage format change, not the int8 head that was
rejected. Halves the head from 106MB to 53MB.
"""
import os, numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper

SRC, OUT = "out/head_fp32.onnx", "out/head_q8.onnx"
for f in (OUT, OUT + ".data"):
    if os.path.exists(f):
        os.remove(f)
m = onnx.load(SRC)
g = m.graph
n = 0
for t in list(g.initializer):
    if t.data_type != TensorProto.FLOAT:
        continue
    a = numpy_helper.to_array(t)
    if a.size < 4096:
        continue
    g.initializer.remove(t)
    g.initializer.append(numpy_helper.from_array(a.astype(np.float16), t.name + "_f16"))
    g.node.insert(0, helper.make_node("Cast", [t.name + "_f16"], [t.name], name=t.name + "_cast",
                                      to=TensorProto.FLOAT))
    n += 1
onnx.save(m, OUT, save_as_external_data=True, location=os.path.basename(OUT) + ".data",
          all_tensors_to_one_file=True, size_threshold=1024)
mm = onnx.load(OUT, load_external_data=False)
for t in mm.graph.initializer:
    for kv in t.external_data:
        if kv.key == "location":
            kv.value = os.path.basename(OUT) + ".data"
onnx.save(mm, OUT)
print("%d tensors -> fp16 storage, %.1f MB" %
      (n, sum(os.path.getsize(f) for f in (OUT, OUT + ".data") if os.path.exists(f)) / 1e6))
