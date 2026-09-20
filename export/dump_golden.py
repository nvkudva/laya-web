"""Golden reference: run the real PyTorch model over the fixtures and dump every
intermediate the ONNX/TS port must reproduce.

Dumps encoder hidden-state stats separately from head logits so a later mismatch
can be localised to one side of the split instead of guessed at.
"""
import json, sys
import numpy as np
import torch

sys.path.insert(0, ".")
from rl_agent_api import RLAgent
from rl_common import QTYPES, build_sequence, collate_items, render_options, temp_bucket

MODEL_DIR = "laya-en"
OUT = "../golden/golden.json"


def main():
    agent = RLAgent(MODEL_DIR, device="cpu")
    tok, cfg, model = agent.tok, agent.cfg, agent.model
    cases = json.load(open("../golden/fixtures.json"))
    out = {"model_dir": MODEL_DIR, "max_len": cfg["max_len"], "head_max_len": cfg["head_max_len"],
           "temperature": cfg["temperature"], "temperature_by_options": cfg["temperature_by_options"],
           "mask_token_id": tok.mask_token_id, "cls_token_id": tok.cls_token_id,
           "sep_token_id": tok.sep_token_id, "pad_token_id": tok.pad_token_id, "cases": []}

    for case in cases:
        rec = {"id": case["id"], "questions": {}}
        for qid, qdef in case["questions"].items():
            q = RLAgent._to_internal(qdef)
            ids, markers = build_sequence(tok, case["state"], q, cfg["max_len"], cfg["head_max_len"])
            k = len(render_options(q))
            qt = QTYPES[q["t"]]
            item = {"ids": ids, "markers": markers, "qtype": qt, "target": [0.0] * len(markers),
                    "label": -1, "episode": 0, "ep_step": 0, "ep_len": 1, "src": "golden"}
            b = collate_items([[item]], tok.pad_token_id)
            with torch.no_grad():
                h = model.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
                logits, act = model(b["input_ids"], b["attention_mask"], b["marker_pos"],
                                    b["marker_mask"], b["qtype"])
            hn = h[0].float().numpy()
            logits = logits[0, :k].float().numpy()
            temp = cfg["temperature_by_options"].get(temp_bucket(qt, k), cfg["temperature"][qt])
            z = logits / temp
            p = np.exp(z - z.max()); p = p / p.sum()
            rec["questions"][qid] = {
                "rendered_options": render_options(q), "qtype": qt, "k": k,
                "temp_bucket": temp_bucket(qt, k), "temperature": float(temp),
                "input_ids": ids, "marker_pos": markers, "n_tokens": len(ids),
                # encoder fingerprint: localises a mismatch to encoder vs head
                "enc_mean": float(hn.mean()), "enc_std": float(hn.std()),
                "enc_cls_head": [float(v) for v in hn[0, :8]],
                "enc_marker0_head": [float(v) for v in hn[markers[0], :8]],
                "logits": [float(v) for v in logits],
                "probs": [float(v) for v in p],
                "act_probs": [float(v) for v in torch.softmax(act[0].float(), -1).numpy()],
            }
        rec["response"] = agent.system_one(case["state"], case["questions"])
        out["cases"].append(rec)
        qs = rec["questions"]
        print("%-28s %s" % (case["id"], "  ".join(
            "%s k=%d L=%d T=%.3f p=[%s]" % (qid, v["k"], v["n_tokens"], v["temperature"],
                                            ",".join("%.3f" % x for x in v["probs"][:4]))
            for qid, v in qs.items())))

    with open(OUT, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print("\nwrote %s (%d cases, %d questions)" % (OUT, len(out["cases"]),
                                                   sum(len(c["questions"]) for c in out["cases"])))


if __name__ == "__main__":
    main()
