# ============================================================
# 📦 sarcasm_utils.py — DeltaWoPerLayerModel inference helper
# ============================================================
import os, math, torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, Any

# ✅ 외부 model_config.py에서 sarcasm 모델 설정 불러오기
from mindcastlib.configs import DefaultModuleConfig


# ============================================================
# 🧩 LoRA Linear + DeltaWoDense
# ============================================================
class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.r = r
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.orthogonal_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(self.dropout(x))) * self.scale


class DeltaWoDense(nn.Module):
    """attention.output.dense 래핑"""
    def __init__(self, base_dense: nn.Linear, hidden_size: int, r: int, alpha: int,
                 dropout: float, delta_scale: float = 1.0, use_cls: bool = True):
        super().__init__()
        self.base_o = base_dense
        for p in self.base_o.parameters():
            p.requires_grad = False
        self.q_lora = LoRALinear(hidden_size, hidden_size, r, alpha, dropout)
        self.k_lora = LoRALinear(hidden_size, hidden_size, r, alpha, dropout)
        self.delta_scale = delta_scale
        self.use_cls = use_cls
        self.hidden_size = hidden_size

    def make_delta(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_lora(x)
        k = self.k_lora(x)
        if self.use_cls:
            qv, kv = q[:, 0, :], k[:, 0, :]
        else:
            qv, kv = q.mean(dim=1), k.mean(dim=1)
        delta = torch.bmm(qv.unsqueeze(2), kv.unsqueeze(1))
        delta = torch.softmax(delta / math.sqrt(qv.size(-1)), dim=-1)
        return delta * self.delta_scale

    def forward(self, x):
        out_base = self.base_o(x)
        delta = self.make_delta(x)
        return out_base + torch.matmul(x, delta)


# ============================================================
# 🧩 Main Model
# ============================================================
class DeltaWoPerLayerModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        base_name = cfg["model_name"]
        self.base_cfg = AutoConfig.from_pretrained(base_name, output_hidden_states=False)
        self.base = AutoModel.from_pretrained(base_name, config=self.base_cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(base_name)

        for p in self.base.parameters():
            p.requires_grad = False

        H = self.base_cfg.hidden_size
        for li, layer in enumerate(self.base.encoder.layer):
            dense = layer.attention.output.dense
            wrapper = DeltaWoDense(
                base_dense=dense,
                hidden_size=H,
                r=8,
                alpha=16,
                dropout=0.1,
                delta_scale=2.0,
                use_cls=True,
            )
            layer.attention.output.dense = wrapper

        self.classifier = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(H // 2, 2),
        )

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = out.last_hidden_state
        cls_vec = hidden[:, 0, :]
        return self.classifier(cls_vec)


# ============================================================
# 🧩 Loader + Inference
# ============================================================
def load_sarcasm_model(device: str = None):
    """model_config.py의 DefaultModuleConfig 기반 sarcasm 모델 로드"""
    cfg = DefaultModuleConfig().sarcasm_model
    device = device or cfg.device

    model = DeltaWoPerLayerModel(cfg.model_dump()).to(device)
    ckpt_path = cfg.ckpt_path
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[INFO] Loading checkpoint from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
    else:
        print(f"[WARN] Checkpoint not found at {ckpt_path}")
    model.eval()
    tokenizer = model.tokenizer
    return model, tokenizer


@torch.no_grad()
def predict_sarcasm(texts, model, tokenizer, device="cuda", max_length=128):
    single = isinstance(texts, str)
    batch_texts = [texts] if single else texts
    enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)
    logits = model(enc["input_ids"], enc["attention_mask"])
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    out = []
    for p, s in zip(preds.cpu().tolist(), probs[:, 1].cpu().tolist()):
        label = "sarcastic" if p == 1 else "non-sarcastic"
        out.append({"label": label, "score": float(s)})
    return out[0] if single else out


# ============================================================
# 🧪 Test Entry
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_sarcasm_model(device=device)

    sample_texts = [
        "아 그럼 이 정부는 진짜 완벽하네요, 실수도 하나 없고!",
        "오늘 날씨가 정말 좋다.",
    ]
    results = predict_sarcasm(sample_texts, model, tokenizer, device=device)
    for t, r in zip(sample_texts, results):
        print(f"🗨️ {t}\n → {r['label']} ({r['score']:.4f})\n")
