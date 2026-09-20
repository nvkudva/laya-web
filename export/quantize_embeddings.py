"""Per-row int8 for the token embedding table.

MatMulNBitsQuantizer leaves the embedding Gather alone, and ORT's dynamic Gather
quantization is per-tensor -- one scale for all 50368 rows -- which measured 0.216
max |dp| on its own. Per-row gives each token its own scale.

Rewrites  Gather(fp32_table, ids)  ->  Cast(Gather(int8_table, ids)) * Gather(scales, ids)
so the full fp32 table is never materialised; only the rows actually used are scaled.
"""
import argparse, os
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="out/enc_nbits8_bs64.onnx")
    ap.add_argument("--out", default="out/encoder_q8.onnx")
    ap.add_argument("--asym", action="store_true")
    ap.add_argument("--block-size", type=int, default=0,
                    help="0 = one scale per row; N = one scale per N-element block within a row")
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
    assert not a.asym or a.block_size, "asym needs a block size"
    assert len(cand) == 1, "expected exactly one embedding Gather, found %d" % len(cand)
    node, tensor = cand[0]
    pre = tensor.name
    W = numpy_helper.to_array(tensor).astype(np.float32)
    print("embedding table %s %s" % (tensor.name, W.shape))

    V, D = W.shape
    bs = a.block_size or D
    assert D % bs == 0, "block size must divide hidden size"
    nblk = D // bs
    Wb = W.reshape(V, nblk, bs)
    # asymmetric (min/max) uses the full 255 codes instead of 254 centred on zero,
    # and costs one extra fp32 per block for the offset.
    if a.asym:
        lo, hi = Wb.min(axis=2), Wb.max(axis=2)
        scale = (hi - lo) / 255.0
        scale[scale == 0] = 1.0
        zp = lo
        Wq = np.clip(np.rint((Wb - zp[:, :, None]) / scale[:, :, None]) - 128, -128, 127).astype(np.int8).reshape(V, D)
        recon = (Wq.reshape(V, nblk, bs).astype(np.float32) + 128) * scale[:, :, None] + zp[:, :, None]
    else:
        scale = np.abs(Wb).max(axis=2) / 127.0            # [V, nblk]
        scale[scale == 0] = 1.0
        zp = None
        Wq = np.clip(np.rint(Wb / scale[:, :, None]), -127, 127).astype(np.int8).reshape(V, D)
        recon = Wq.reshape(V, nblk, bs).astype(np.float32) * scale[:, :, None]
    err = np.abs(recon - Wb).max()
    mean_err = float(np.abs(recon - Wb).mean())
    print("int8 emb, %d scale(s)/row: max %.3e mean %.3e recon error (table absmax %.3f)"
          % (nblk, err, mean_err, np.abs(W).max()))

    g.initializer.remove(tensor)
    g.initializer.extend([
        numpy_helper.from_array(Wq, tensor.name + "_i8"),
        numpy_helper.from_array(scale.astype(np.float32), tensor.name + "_scale"),
    ])
    if nblk > 1:
        g.initializer.extend([
            numpy_helper.from_array(np.array([0, 0, nblk, bs], dtype=np.int64), pre + "_shp4"),
            numpy_helper.from_array(np.array([0, 0, D], dtype=np.int64), pre + "_shp3"),
        ])

    out = node.output[0]
    ids = node.input[1]
    new = [
        helper.make_node("Gather", [tensor.name + "_i8", ids], [pre + "_gq"], name=pre + "_g_i8", axis=0),
        helper.make_node("Cast", [pre + "_gq"], [pre + "_gf"], name=pre + "_cast", to=TensorProto.FLOAT),
        helper.make_node("Gather", [tensor.name + "_scale", ids], [pre + "_gs"], name=pre + "_g_scale", axis=0),
    ]
    if nblk == 1:
        new.append(helper.make_node("Mul", [pre + "_gf", pre + "_gs"], [out], name=pre + "_descale"))
    else:
        new += [
            helper.make_node("Reshape", [pre + "_gf", pre + "_shp4"], [pre + "_gf4"], name=pre + "_rs4", allowzero=0),
            helper.make_node("Unsqueeze", [pre + "_gs", pre + "_ax"], [pre + "_gs4"], name=pre + "_unsq"),
            helper.make_node("Mul", [pre + "_gf4", pre + "_gs4"], [pre + "_m4"], name=pre + "_descale"),
            helper.make_node("Reshape", [pre + "_m4", pre + "_shp3"], [out], name=pre + "_rs3", allowzero=0),
        ]
        g.initializer.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), pre + "_ax"))
    idx = list(g.node).index(node)
    g.node.remove(node)
    for i, n in enumerate(new):
        g.node.insert(idx + i, n)

    onnx.save(m, a.out, save_as_external_data=True, location=os.path.basename(a.out) + ".data",
              all_tensors_to_one_file=True, size_threshold=1024)
    tot = sum(os.path.getsize(f) for f in (a.out, a.out + ".data") if os.path.exists(f))
    print("%-24s %8.1f MB" % (os.path.basename(a.out), tot / 1e6))


if __name__ == "__main__":
    main()
