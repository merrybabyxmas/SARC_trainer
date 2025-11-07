# ============================================================
# 🎯 script/train.py — main entry (run directly)
# ============================================================

import os, sys, csv, random, torch, numpy as np, matplotlib.pyplot as plt
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

# --- 경로 설정 (상위 폴더 import 가능하게) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.models import DeltaWoPerLayerModel
from utils.dataset import SarcasmDataset
from utils.logger import logger
from config.TRAINERCONFIG import CFG


# ---------------- Utils ----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, ce):
    model.eval()
    preds, labels, total_loss = [], [], 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = ce(logits, batch["labels"])
        total_loss += loss.item()
        preds.extend(logits.argmax(-1).cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return total_loss / len(loader), acc, p, r, f1


def train(cfg=CFG):
    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)
    set_seed(cfg["seed"])
    device = cfg["device"]

    logger.info(f"🚀 Starting training on {device}")
    model = DeltaWoPerLayerModel(cfg).to(device)
    tokenizer = model.tokenizer

    tr_loader = DataLoader(
        SarcasmDataset(cfg["data"]["train_csv"], tokenizer, cfg["data"]["max_length"]),
        batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], pin_memory=True
    )
    va_loader = DataLoader(
        SarcasmDataset(cfg["data"]["val_csv"], tokenizer, cfg["data"]["max_length"]),
        batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    total_steps = len(tr_loader) * cfg["train"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(cfg["train"]["warmup_ratio"] * total_steps),
        total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["fp16"])
    ce = nn.CrossEntropyLoss(label_smoothing=cfg["loss"]["label_smoothing"]).to(device)

    best_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "acc": [], "precision": [], "recall": [], "f1": []}

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(tr_loader):
            batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}
            with torch.cuda.amp.autocast(enabled=cfg["train"]["fp16"]):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = ce(logits, batch["labels"])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            total_loss += loss.item()

            if step % 50 == 0:
                logger.info(f"[Epoch {epoch}] Step {step}/{len(tr_loader)} | Loss={loss.item():.4f}")

        val_loss, acc, p, r, f1 = evaluate(model, va_loader, device, ce)
        history["train_loss"].append(total_loss / len(tr_loader))
        history["val_loss"].append(val_loss)
        history["acc"].append(acc); history["precision"].append(p)
        history["recall"].append(r); history["f1"].append(f1)

        logger.info(f"[Epoch {epoch}] Train={total_loss/len(tr_loader):.4f} | Val={val_loss:.4f} | F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(cfg["train"]["save_dir"], "best.pt"))
            logger.info(f"💾 Best model saved → F1={best_f1:.4f}")

    # Save curve & CSV
    epochs = range(1, len(history["f1"]) + 1)
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(epochs, history["train_loss"], "--", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["f1"], "g-", label="F1")
    ax2.plot(epochs, history["acc"], "r-", label="Acc")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax2.set_ylabel("Score")
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.tight_layout()
    out_png = os.path.join(cfg["train"]["save_dir"], "training_curve.png")
    plt.savefig(out_png)
    plt.close(fig)

    csv_path = os.path.join(cfg["train"]["save_dir"], "training_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch","train_loss","val_loss","acc","precision","recall","f1"])
        for i in range(len(history["f1"])):
            w.writerow([i+1, history["train_loss"][i], history["val_loss"][i],
                        history["acc"][i], history["precision"][i],
                        history["recall"][i], history["f1"][i]])
    logger.info(f"📊 Saved curve → {out_png}")
    logger.info(f"📄 Saved log → {csv_path}")
    logger.info("✅ Training completed.")


# ---------------- Entry ----------------
if __name__ == "__main__":
    train(CFG)
