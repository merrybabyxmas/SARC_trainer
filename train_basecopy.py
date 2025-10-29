import os
import math
import yaml
import time
import random
import logging
from typing import Optional, Dict, Any, Tuple, List
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup
)





# ============================================================
# 🧱 Global settings
# ============================================================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


# ============================================================
# 🧾 Logger
# ============================================================
def setup_logger():
    logger = logging.getLogger("SarcDetector")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


logger = setup_logger()


# ============================================================
# ⚙️ Config
# ============================================================
DEFAULT_CFG = {
    "seed": 42,
    "device": "cuda:2" if torch.cuda.is_available() else "cpu",
    "model": {
        "base_model_name": "bert-base-uncased",
        "freeze_base": True,
        "sarc_layers": 10,
        "hidden_drop": 0.3,
        "alpha_merge": 0.5,
        "use_base_attn_logits": False,
    },
    "data": {
        "train_csv": "/home/dongwoo44/data/sarc/train_bal_textlabel.csv",
        "val_csv": "/home/dongwoo44/data/sarc/test_bal_textlabel.csv",
        "max_length": 256,
        "batch_size": 64,
        "num_workers": 2,
    },
    "train": {
        "epochs": 500,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "log_interval": 50,
        "save_dir": "./checkpoints",
        "save_interval": 50,
        "gradient_clip": 0.9,
        "fp16": True
    }
}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml_or_default(path: Optional[str]) -> Dict[str, Any]:
    if path and os.path.isfile(path):
        with open(path, "r") as f:
            user = yaml.safe_load(f)
        def deep_merge(a, b):
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    deep_merge(a[k], v)
                else:
                    a[k] = v
            return a
        return deep_merge(DEFAULT_CFG.copy(), user or {})
    return DEFAULT_CFG.copy()


import csv



# ============================================================
# 📚 Dataset
# ============================================================
class SarcDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: AutoTokenizer, max_length: int):
        super().__init__()
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)

        with open(csv_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if i == 0 and ("text" in line and "label" in line):
                    continue
                if "," in line:
                    text, label = line.rsplit(",", 1)
                    self.samples.append((text.strip(), int(label.strip())))

        if not self.samples:
            raise ValueError(f"No valid samples found in {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# ============================================================
# 🧩 Sarcasm Blocks (Stable Version)
# ============================================================

# ============================================================
# 🧩 Sarcasm Blocks (return (B,H,S,S))
# ============================================================
class SarcPreAttentionBlock(nn.Module):
    """Stable pre-attention block producing (B,H,S,S)"""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.3, expansion: int = 4):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # lightweight gating MLP
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.head_dim),
            nn.Linear(self.head_dim, expansion * self.head_dim),
            nn.ReLU(),
            nn.Linear(expansion * self.head_dim, self.head_dim),
            nn.Tanh()
        )

    def _shape_heads(self, x):
        B, S, D = x.shape
        H, d = self.num_heads, self.head_dim
        return x.view(B, S, H, d).transpose(1, 2)  # (B,H,S,d)

    def forward(self, hidden_states):
        q = self._shape_heads(self.q_proj(hidden_states))
        k = self._shape_heads(self.k_proj(hidden_states))

        # apply small nonlinear modulation
        q_mod = self.mlp(q)

        logits = torch.matmul(q_mod, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logits = torch.tanh(logits)  # ensure [-1,1]
        logits = self.dropout(logits)
        return logits  # (B,H,S,S)


class SarcStack(nn.Module):
    """Stacked sarcasm attention maps producing (B,H,S,S)"""
    def __init__(self, hidden_size, num_heads, depth=4, dropout=0.3):
        super().__init__()
        self.blocks = nn.ModuleList([
            SarcPreAttentionBlock(hidden_size, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        attn_accum = 0
        for blk in self.blocks:
            A = blk(hidden_states)  # (B,H,S,S)
            attn_accum = attn_accum + A
            # simple residual update for feature refinement
            feedback = torch.mean(A, dim=1)  # (B,S,S)
            hidden_states = hidden_states + torch.bmm(feedback, hidden_states)
            hidden_states = self.norm(hidden_states)
        return torch.tanh(attn_accum / len(self.blocks))  # (B,H,S,S)


# ============================================================
# 🧠 Model
# ============================================================



class SarcModuleFromBase(nn.Module):
    """
    Sarcasm module that replicates base model's attention structure
    but replaces softmax with [-1,1] normalization.
    """
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "q_proj": nn.Linear(hidden_size, hidden_size),
                "k_proj": nn.Linear(hidden_size, hidden_size),
                "v_proj": nn.Linear(hidden_size, hidden_size),
                "o_proj": nn.Linear(hidden_size, hidden_size),
                "norm": nn.LayerNorm(hidden_size),
                "drop": nn.Dropout(dropout)
            })
            for _ in range(num_layers)
        ])

    def _shape_heads(self, x: torch.Tensor):
        B, S, D = x.shape
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,S,d)

    def _merge_heads(self, x: torch.Tensor):
        B, H, S, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, H * d)  # (B,S,D)

    def forward(self, hidden_states: torch.Tensor):
        """
        hidden_states: (B, S, D)
        return: (B, S, D)
        """
        for layer in self.layers:
            q = self._shape_heads(layer["q_proj"](hidden_states))
            k = self._shape_heads(layer["k_proj"](hidden_states))
            v = self._shape_heads(layer["v_proj"](hidden_states))

            # base attention logits
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # ✅ [-1,1] scaling instead of softmax
            min_val = attn_logits.min(dim=-1, keepdim=True)[0]
            max_val = attn_logits.max(dim=-1, keepdim=True)[0]
            attn_scaled = 2 * (attn_logits - min_val) / (max_val - min_val + 1e-6) - 1

            attn_scaled = layer["drop"](attn_scaled)

            # apply attention
            context = torch.matmul(attn_scaled, v)
            context = self._merge_heads(context)
            context = layer["o_proj"](context)

            # residual & norm
            hidden_states = layer["norm"](hidden_states + context)

        return hidden_states

class SarcDetection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        base_name = cfg["model"]["base_model_name"]

        self.base_cfg = AutoConfig.from_pretrained(base_name, output_attentions=True, output_hidden_states=True)
        self.base = AutoModel.from_pretrained(base_name, config=self.base_cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(base_name)

        H = self.base_cfg.hidden_size
        Hh = self.base_cfg.num_attention_heads
        self.num_heads = Hh
        self.head_dim = H // Hh

        # Sarcasm module produces (B, H, S, S)
        self.sarc_module = SarcModuleFromBase(
            hidden_size=self.base_cfg.hidden_size,
            num_heads=self.base_cfg.num_attention_heads,
            num_layers=cfg["model"]["sarc_layers"],
            dropout=cfg["model"]["hidden_drop"]
        )
        # Projection
        self.proj = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H // 2),
            nn.ReLU(),
            nn.Linear(H // 2, 2)
        )

        # Extract final attention weights from BERT
        last_layer = self.base.encoder.layer[-1]
        self.W_q = last_layer.attention.self.query
        self.W_k = last_layer.attention.self.key
        self.W_v = last_layer.attention.self.value
        self.W_o = last_layer.attention.output.dense

        # Freeze policy
        if cfg["model"]["freeze_base"]:
            for p in self.base.parameters():
                p.requires_grad = False
            for name, p in self.base.named_parameters():
                if any(k in name for k in ["encoder.layer.10", "encoder.layer.11"]):
                    p.requires_grad = True
            logger.info("🧊 Base encoder frozen (except last 2 layers).")

        # Param summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total: {total_params:,} | Trainable: {trainable_params:,}")

    def _shape_heads(self, x):
        B, S, D = x.shape
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        B, H, S, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, H * d)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True
        )

        hidden = outputs.last_hidden_state  # (B, S, D)
        q = self._shape_heads(self.W_q(hidden))  # (B,H,S,d)
        k = self._shape_heads(self.W_k(hidden))  # (B,H,S,d)
        v = self._shape_heads(self.W_v(hidden))  # (B,H,S,d)

        # ---- Base attention ----
        A_base_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        A_base = F.softmax(A_base_logits, dim=-1)

        # ---- Sarcasm attention ----
        A_sarc = self.sarc_stack(hidden)  # (B,H,S,S)

        # ---- Fusion ----
        A_fused = A_base * A_sarc  # elementwise (B,H,S,S)

        # ---- Apply to Value ----
        context = torch.matmul(A_fused, v)  # (B,H,S,d)
        context = self._merge_heads(context)  # (B,S,D)
        context = self.W_o(context)

        # ---- Classification ----
        cls_vec = context[:, 0, :]
        logits = self.proj(cls_vec)
        return logits


# ============================================================
# 🏋️ Training / Evaluation
# ============================================================
def evaluate(model: SarcDetection, loader: DataLoader, device: str, epoch: int, save_dir: str) -> Tuple[float, float]:
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    results = []

    with torch.no_grad():
        for batch in loader:
            # ✅ GPU -> CPU 변환 후 디코딩
            input_ids_cpu = batch["input_ids"].cpu().tolist()
            texts = loader.dataset.tokenizer.batch_decode(input_ids_cpu, skip_special_tokens=True)

            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids", None)
            )

            loss = criterion(logits, batch["labels"])
            preds = logits.argmax(dim=-1)
            correct_batch = (preds == batch["labels"]).long().tolist()

            for t, y_true, y_pred, corr in zip(texts, batch["labels"].tolist(), preds.tolist(), correct_batch):
                results.append({
                    "text": t.replace("\n", " ")[:300],  # 줄바꿈 제거 및 300자 제한
                    "label": y_true,
                    "pred": y_pred,
                    "correct": corr
                })

            correct += sum(correct_batch)
            total += preds.numel()
            loss_sum += loss.item() * preds.numel()

    acc = correct / max(1, total)
    avg_loss = loss_sum / max(1, total)

    # ✅ Save predictions
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"val_results_epoch{epoch}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "pred", "correct"])
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"🧾 Saved validation predictions → {csv_path}")
    return acc, avg_loss

def train(config_path: Optional[str] = None):
    cfg = load_yaml_or_default(config_path)
    set_seed(cfg["seed"])
    device = cfg["device"]

    logger.info("🚀 Initializing Sarcasm Detection model...")
    model = SarcDetection(cfg).to(device)
    tokenizer = model.tokenizer

    train_loader, val_loader = build_loaders(cfg, tokenizer)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(cfg["train"]["warmup_ratio"] * len(train_loader) * cfg["train"]["epochs"]),
        len(train_loader) * cfg["train"]["epochs"]
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["fp16"])
    criterion = nn.CrossEntropyLoss()

    log_dir = cfg["train"]["save_dir"]
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train_log.csv")

    # ✅ CSV 로그 헤더 초기화
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])

    logger.info(f"Training on {device} for {cfg['train']['epochs']} epochs")

    best_val_loss = float("inf")
    best_acc = 0.0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=cfg["train"]["fp16"]):
                logits = model(**{k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids"] if k in batch})
                loss = criterion(logits, batch["labels"])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()

            if step % cfg["train"]["log_interval"] == 0:
                logger.info(f"[Epoch {epoch}] Step {step}/{len(train_loader)} | loss={running_loss/(step+1):.4f}")

        # ✅ 평균 train loss
        train_loss = running_loss / len(train_loader)

        # ✅ 매 epoch마다 validation 수행
        val_acc, val_loss = evaluate(model, val_loader, device, epoch, cfg["train"]["save_dir"])
        logger.info(f"[Epoch {epoch}] ✅ Validation | val_acc={val_acc:.4f}, val_loss={val_loss:.4f}")

        # ✅ 로그 저장
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_acc])

        # ✅ best val_loss 갱신 시 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_acc = val_acc
            best_path = os.path.join(cfg["train"]["save_dir"], "best.pt")
            torch.save(model.state_dict(), best_path)
            logger.info(f"💾 New best model saved → {best_path} | val_loss={best_val_loss:.4f}, val_acc={best_acc:.4f}")

    logger.info("🎯 Training complete.")
    logger.info(f"📊 Full training log saved to {log_path}")

def build_loaders(cfg, tokenizer):
    train_ds = SarcDataset(cfg["data"]["train_csv"], tokenizer, cfg["data"]["max_length"])
    val_ds = SarcDataset(cfg["data"]["val_csv"], tokenizer, cfg["data"]["max_length"])
    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=2)
    return train_loader, val_loader


# ============================================================
# 🚀 Entry
# ============================================================
if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    train(config_path)
