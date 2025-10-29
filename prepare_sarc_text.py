import os
import bz2
import json
import pandas as pd
from tqdm import tqdm

DATA_DIR = "./data/sarc"
COMMENTS_PATH = os.path.join(DATA_DIR, "comments.json.bz2")
TRAIN_PATH = os.path.join(DATA_DIR, "train-balanced.csv.bz2")
TEST_PATH = os.path.join(DATA_DIR, "test-balanced.csv.bz2")

# ==========================================================
# 1️⃣ Load the full JSON dictionary (not line-by-line)
# ==========================================================
def load_comments_dict(path):
    print(f"🔍 Loading comments.json.bz2 ...")
    with bz2.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    comments = {cid: obj["text"] for cid, obj in data.items() if "text" in obj}
    print(f"✅ Loaded {len(comments):,} comments.")
    return comments


# ==========================================================
# 2️⃣ Build paired-text dataset (parent + reply)
# ==========================================================
def build_text_dataset(csv_path, comments, out_path):
    print(f"📖 Building text dataset from {csv_path} ...")
    rows = []
    with bz2.open(csv_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            _, pair, label = parts

            # ✅ Fix: handle "1 0" cases safely
            label_token = label.strip().split()[0]
            try:
                label = int(label_token)
            except ValueError:
                continue

            ids = pair.strip().split()
            if len(ids) != 2:
                continue

            parent_id, child_id = ids
            parent = comments.get(parent_id)
            child = comments.get(child_id)
            if not parent or not child:
                continue

            text = f"[PARENT]: {parent} [REPLY]: {child}"
            rows.append((text, label))

    df = pd.DataFrame(rows, columns=["text", "label"])
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Saved {len(df):,} samples → {out_path}")

# ==========================================================
# 3️⃣ Main
# ==========================================================
if __name__ == "__main__":
    comments = load_comments_dict(COMMENTS_PATH)
    build_text_dataset(TRAIN_PATH, comments, os.path.join(DATA_DIR, "train_bal_textlabel.csv"))
    build_text_dataset(TEST_PATH, comments, os.path.join(DATA_DIR, "test_bal_textlabel.csv"))
