"""Compare an ONNX encoder/head pair against golden.json.

Reports encoder drift and head drift separately, then the metrics the parity gate
is defined on: argmax agreement, max |dp|, mean KL(p_ref || p_onnx).
"""
import argparse, json
import numpy as np
import onnxruntime as ort

GOLD = "../golden/golden.json"


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def run(enc_path, head_path, gold):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    enc = ort.InferenceSession(enc_path, so, providers=["CPUExecutionProvider"])
    head = ort.InferenceSession(head_path, so, providers=["CPUExecutionProvider"])
    rows = []
    for case in gold["cases"]:
        for qid, g in case["questions"].items():
            ids = np.array([g["input_ids"]], dtype=np.int64)
            att = np.ones_like(ids)
            mpos = np.array([g["marker_pos"]], dtype=np.int64)
            mmask = np.ones((1, len(g["marker_pos"])), dtype=bool)
            qt = np.array([g["qtype"]], dtype=np.int64)
            h = enc.run(["hidden"], {"input_ids": ids, "attention_mask": att})[0]
            logits, act = head.run(["logits", "act_logits"],
                                   {"hidden": h, "attention_mask": att, "marker_pos": mpos,
                                    "marker_mask": mmask, "qtype": qt})
            k = g["k"]
            lg = logits[0, :k].astype(np.float64)
            p = softmax(lg / g["temperature"])
            pr = np.array(g["probs"], dtype=np.float64)
            rows.append({
                "case": "%s/%s" % (case["id"], qid), "k": k,
                "d_enc_mean": abs(float(h[0].mean()) - g["enc_mean"]),
                "d_enc_cls": float(np.abs(h[0, 0, :8] - np.array(g["enc_cls_head"])).max()),
                "d_logit": float(np.abs(lg - np.array(g["logits"])).max()),
                "d_p": float(np.abs(p - pr).max()),
                "kl": float((pr * np.log(np.clip(pr, 1e-12, None) / np.clip(p, 1e-12, None))).sum()),
                "argmax_ok": int(p.argmax()) == int(pr.argmax()),
                "d_act": float(np.abs(softmax(act[0].astype(np.float64)) - np.array(g["act_probs"])).max()),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="out/encoder_fp32.onnx")
    ap.add_argument("--head", default="out/head_fp32.onnx")
    ap.add_argument("--gate", action="store_true", help="apply the int8 acceptance gate")
    a = ap.parse_args()
    gold = json.load(open(GOLD))
    rows = run(a.encoder, a.head, gold)

    worst = sorted(rows, key=lambda r: -r["d_p"])[:6]
    print("%-34s %3s %10s %10s %9s %9s %6s" % ("case", "k", "d_enc_cls", "d_logit", "d_p", "kl", "argmx"))
    for r in worst:
        print("%-34s %3d %10.2e %10.2e %9.2e %9.2e %6s" % (r["case"], r["k"], r["d_enc_cls"],
                                                           r["d_logit"], r["d_p"], r["kl"], r["argmax_ok"]))
    n = len(rows)
    agree = sum(r["argmax_ok"] for r in rows) / n
    max_dp = max(r["d_p"] for r in rows)
    mean_kl = sum(r["kl"] for r in rows) / n
    max_act = max(r["d_act"] for r in rows)
    print("\n%d questions | argmax %.1f%% | max |dp| %.3e | mean KL %.3e | max |d act| %.3e"
          % (n, agree * 100, max_dp, mean_kl, max_act))
    print("encoder: max |d cls| %.3e" % max(r["d_enc_cls"] for r in rows))

    if a.gate:
        checks = [("argmax >= 99%", agree >= 0.99, "%.1f%%" % (agree * 100)),
                  ("max |dp| <= 0.02", max_dp <= 0.02, "%.4f" % max_dp),
                  ("mean KL <= 1e-3", mean_kl <= 1e-3, "%.2e" % mean_kl)]
        print()
        for name, ok, val in checks:
            print("  %-20s %-8s %s" % (name, "PASS" if ok else "FAIL", val))
        raise SystemExit(0 if all(c[1] for c in checks) else 1)


if __name__ == "__main__":
    main()
