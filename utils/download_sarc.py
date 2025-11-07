# ============================================================
# 📦 prepare_sarc_dataset.py
#  - Download + Extract SARC 2.0 (political)
#  - Parse comments.json.bz2 to build parent-reply text dataset
#  - Save train_bal_textlabel.csv / test_bal_textlabel.csv
# ============================================================

import os
import urllib.request
import bz2
import shutil
import json
import pandas as pd
from tqdm import tqdm

# ============================================================
# ⚙️ CONFIG
# ============================================================
DATA_DIR = "./test/sarc"
BASE_URL = "https://nlp.cs.princeton.edu/old/SARC/2.0/pol/"
CSV_FILES = ["train-balanced.csv.bz2", "test-balanced.csv.bz2"]
COMMENT_FILE = "comments.json.bz2"


# ============================================================
# 1️⃣ Download & Extract
# ============================================================
def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)
    for csv_file in CSV_FILES:
        csv_path = os.path.join(DATA_DIR, csv_file)
        extracted_path = csv_path[:-4]  # remove .bz2

        # Download
        if not os.path.exists(csv_path):
            print(f"⬇️ Downloading {csv_file} ...")
            urllib.request.urlretrieve(BASE_URL + csv_file, csv_path)
        else:
            print(f"📦 {csv_file} already exists, skipping.")

        # Extract
        if not os.path.exists(extracted_path):
            print(f"🔧 Extracting {csv_file} ...")
            with bz2.open(csv_path, "rb") as f_in, open(extracted_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            print(f"✅ Extracted → {extracted_path}")
        else:
            print(f"📂 {extracted_path} already extracted.")

    # Download comments.json.bz2
    comments_path = os.path.join(DATA_DIR, COMMENT_FILE)
    if not os.path.exists(comments_path):
        print(f"⬇️ Downloading {COMMENT_FILE} ...")
        urllib.request.urlretrieve(BASE_URL + COMMENT_FILE, comments_path)
        print(f"✅ Downloaded → {comments_path}")
    else:
        print(f"📂 {COMMENT_FILE} already exists, skipping.")

    print("\n📦 All raw files ready!")
    return (
        os.path.join(DATA_DIR, "train-balanced.csv.bz2"),
        os.path.join(DATA_DIR, "test-balanced.csv.bz2"),
        os.path.join(DATA_DIR, "comments.json.bz2"),
    )


# ============================================================
# 2️⃣ Load comments.json (id → text)
# ============================================================
def load_comments_dict(path):
    print(f"🔍 Loading comments.json.bz2 ... (this may take a minute)")
    with bz2.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    comments = {cid: obj["text"] for cid, obj in data.items() if "text" in obj}
    print(f"✅ Loaded {len(comments):,} comments.")
    return comments


# ============================================================
# 3️⃣ Build paired text dataset
# ============================================================
def build_text_dataset(csv_bz2_path, comments, out_path):
    print(f"🧩 Building text dataset from {csv_bz2_path} ...")
    rows = []
    with bz2.open(csv_bz2_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            _, pair, label_str = parts

            label_token = label_str.strip().split()[0]
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


# ============================================================
# 4️⃣ Full pipeline
# ============================================================
def prepare_sarc_balanced_dataset():
    print("🚀 Starting SARC balanced dataset preparation...")
    train_csv_bz2, test_csv_bz2, comments_bz2 = download_and_extract()
    comments = load_comments_dict(comments_bz2)

    train_out = os.path.join(DATA_DIR, "train_bal_textlabel.csv")
    test_out = os.path.join(DATA_DIR, "test_bal_textlabel.csv")

    build_text_dataset(train_csv_bz2, comments, train_out)
    build_text_dataset(test_csv_bz2, comments, test_out)

    print("\n🎯 All done! Final datasets ready:")
    print(f"   📁 {train_out}")
    print(f"   📁 {test_out}")
    return train_out, test_out


# ============================================================
# 🧩 Entry Point
# ============================================================
if __name__ == "__main__":
    prepare_sarc_balanced_dataset()
