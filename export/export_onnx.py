"""Export Laya as two ONNX graphs: encoder (quantizable) and head (stays fp32).

Split on purpose -- one fused graph makes an export failure undiagnosable, and the
head must stay fp32 because this checkpoint's fitted temperatures were computed on
fp32 logits.
"""
import argparse, json, os, sys
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from rl_common import load_cfg, build_model

MODEL_DIR = "laya-en"
OUT_DIR = "out"


class HeadWrapper(nn.Module):
    """Everything after the encoder: type_emb + 2 head layers + marker gather + scorer + act head.

    Mirrors DecisionModel.forward from the encoder output onward, byte for byte.
    """

    def __init__(self, m):
        super().__init__()
        self.type_emb, self.head, self.scorer, self.act_head = m.type_emb, m.head, m.scorer, m.act_head

    def forward(self, hidden, attention_mask, marker_pos, marker_mask, qtype):
        h = hidden + self.type_emb(qtype)[:, None, :]
        pad = ~attention_mask.bool()
        for layer in self.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits, -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        act_logits = self.act_head(torch.cat([h[:, 0].float(), feats], -1))
        return logits, act_logits


class EncoderWrapper(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, input_ids, attention_mask):
        return self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def export(dynamo: bool):
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = load_cfg(os.path.join(MODEL_DIR, "rl_agent_config.json"))
    model = build_model(cfg, os.path.join(MODEL_DIR, "encoder"))
    from safetensors.torch import load_file
    model.load_state_dict(load_file(os.path.join(MODEL_DIR, "model.safetensors")), strict=True)
    model.eval()
    model.encoder.config.reference_compile = False
    d = model.encoder.config.hidden_size

    # nn.TransformerEncoderLayer takes a fused fastpath under eval+no_grad that does not trace.
    torch.backends.mha.set_fastpath_enabled(False)

    L, K = 48, 3
    ids = torch.randint(100, 5000, (1, L), dtype=torch.long)
    att = torch.ones((1, L), dtype=torch.long)
    hid = torch.randn(1, L, d)
    mpos = torch.tensor([[3, 9, 15]], dtype=torch.long)
    mmask = torch.ones((1, K), dtype=torch.bool)
    qtype = torch.tensor([0], dtype=torch.long)

    # opset 18: the exporter emits opset-18 Split (num_outputs attribute) regardless of
    # what it declares, so declaring 17 produces a graph ORT rejects as invalid.
    kw = {"opset_version": 18, "dynamo": dynamo}

    enc_path = os.path.join(OUT_DIR, "encoder_fp32.onnx")
    print("exporting encoder -> %s (dynamo=%s)" % (enc_path, dynamo))
    torch.onnx.export(
        EncoderWrapper(model.encoder), (ids, att), enc_path,
        input_names=["input_ids", "attention_mask"], output_names=["hidden"],
        dynamic_axes={"input_ids": {0: "b", 1: "L"}, "attention_mask": {0: "b", 1: "L"},
                      "hidden": {0: "b", 1: "L"}}, **kw)

    head_path = os.path.join(OUT_DIR, "head_fp32.onnx")
    print("exporting head -> %s" % head_path)
    torch.onnx.export(
        HeadWrapper(model), (hid, att, mpos, mmask, qtype), head_path,
        input_names=["hidden", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_logits"],
        dynamic_axes={"hidden": {0: "b", 1: "L"}, "attention_mask": {0: "b", 1: "L"},
                      "marker_pos": {0: "b", 1: "K"}, "marker_mask": {0: "b", 1: "K"},
                      "qtype": {0: "b"}, "logits": {0: "b", 1: "K"}, "act_logits": {0: "b"}}, **kw)

    for p in (enc_path, head_path):
        print("%-28s %8.1f MB" % (os.path.basename(p), os.path.getsize(p) / 1e6))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamo", action="store_true", default=True)
    export(ap.parse_args().dynamo)
