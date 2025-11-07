# ============================================================
# 📦 core/models.py
#  - DeltaWoDense (per-layer) + LoRA-style ΔW_o 생성 구조
# ============================================================

import math, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, Any
from utils.logger import logger


# ---------------- LoRA Linear ----------------
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


# ---------------- Dense Wrapper ----------------
class DeltaWoDense(nn.Module):
    """
    원래 attention.output.dense를 래핑하여
    out = base_o(x) + x @ ΔW_o 를 수행.
    """
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


# ---------------- Main Model ----------------
class DeltaWoPerLayerModel(nn.Module):
    """
    DeBERTa-v3-large + 각 Attention Layer마다 ΔWₒ 부착.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        base_name = cfg["model"]["base_name"]
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
                r=cfg["model"]["r"],
                alpha=cfg["model"]["alpha"],
                dropout=cfg["model"]["dropout"],
                delta_scale=cfg["model"]["delta_scale"],
                use_cls=cfg["model"]["use_cls"],
            )
            layer.attention.output.dense = wrapper
            n_params = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
            logger.info(f"  🩶 ΔWₒ attached → layer {li:02d} | trainable={n_params:,}")

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
