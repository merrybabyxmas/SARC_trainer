# ============================================================
# 📄 TRAINERCONFIG.py
# ============================================================

import torch

CFG = {
    "seed": 42,
    "device": "cuda:1" if torch.cuda.is_available() else "cpu",
    "model": {
        "base_name": "microsoft/deberta-v3-large",
        "r": 8,
        "alpha": 16,
        "dropout": 0.10,
        "delta_scale": 2.0,
        "use_cls": True
    },
    "train": {
        "epochs": 300,
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "fp16": True,
        "gradient_clip": 1.0,
        "save_dir": "./checkpoints_deltaWo_per_layer",
        "grad_accum": 1
    },
    "data": {
        "train_csv": "/home/dongwoo44/data/sarc/train_bal_textlabel.csv",
        "val_csv":   "/home/dongwoo44/data/sarc/test_bal_textlabel.csv",
        "max_length": 256,
        "num_workers": 2,
    },
    "loss": {
        "label_smoothing": 0.0
    }
}
