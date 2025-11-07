# ============================================================
# 📚 utils/dataset.py
# ============================================================

import torch
from torch.utils.data import Dataset

class SarcasmDataset(Dataset):
    """
    CSV: text,label 형식
    """
    def __init__(self, csv_path, tokenizer, max_length):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or "text" in line:
                    continue
                text, label = line.rsplit(",", 1)
                self.data.append((text.strip(), int(label.strip())))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        enc = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label)
        item["raw_text"] = text
        return item
