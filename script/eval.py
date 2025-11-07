# ============================================================
# 🎯 script/eval.py — Evaluate trained model (.pt) using config
# ============================================================

import os, sys, csv, torch, numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 상위 폴더 import 가능하게 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.models import DeltaWoPerLayerModel
from utils.dataset import SarcasmDataset
from utils.logger import logger
from config.EVALCONFIG import EVAL_CFG
from config.TRAINERCONFIG import CFG as TRAIN_CFG  # 모델 로딩용 구조


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    preds, labels, probs = [], [], []
    for batch in tqdm(loader, desc="Evaluating", ncols=90):
        batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}
        logits = model(batch["input_ids"], batch["attention_mask"])
        prob = torch.softmax(logits, dim=-1)
        preds.extend(prob.argmax(dim=-1).cpu().tolist())
        probs.extend(prob[:, 1].cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
    return np.array(preds), np.array(labels), np.array(probs)


def save_results(preds, labels, probs, texts, save_dir):
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    cm = confusion_matrix(labels, preds)

    logger.info(f"✅ Evaluation results — ACC={acc:.4f}, P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    os.makedirs(save_dir, exist_ok=True)

    # summary.txt
    with open(os.path.join(save_dir, "eval_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {p:.4f}\nRecall: {r:.4f}\nF1: {f1:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        np.savetxt(f, cm, fmt="%d")

    # predictions.csv
    csv_path = os.path.join(save_dir, "eval_predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "pred", "prob"])
        for t, l, p_, pb in zip(texts, labels, preds, probs):
            w.writerow([t, l, p_, f"{pb:.4f}"])

    logger.info(f"📄 Saved predictions → {csv_path}")
    logger.info(f"📊 Saved summary → {os.path.join(save_dir, 'eval_summary.txt')}")


def main():
    # Load config values
    cfg = EVAL_CFG
    device = cfg["device"]
    ckpt_path = cfg["paths"]["ckpt_path"]
    test_csv = cfg["paths"]["test_csv"]
    save_dir = cfg["paths"]["save_dir"]

    os.makedirs(save_dir, exist_ok=True)

    # --- Load model ---
    logger.info(f"🧩 Loading model from {ckpt_path}")
    model = DeltaWoPerLayerModel(TRAIN_CFG).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    tokenizer = model.tokenizer

    # --- Dataset ---
    logger.info(f"📂 Loading test data from {test_csv}")
    dataset = SarcasmDataset(test_csv, tokenizer, cfg["data"]["max_length"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    # --- Evaluate ---
    preds, labels, probs = evaluate_model(model, loader, device)
    texts = [d[0] for d in dataset.data]

    # --- Save Results ---
    save_results(preds, labels, probs, texts, save_dir)


if __name__ == "__main__":
    main()
