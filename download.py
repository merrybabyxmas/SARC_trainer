import os
import urllib.request
import bz2
import shutil
import pandas as pd

def download_sarc_balanced_basic(save_dir: str = "./data/sarc"):
    """
    Download and extract SARC 2.0 political balanced train/test sets
    along with comments.json.bz2 (the actual text data).
    """

    os.makedirs(save_dir, exist_ok=True)

    base_url = "https://nlp.cs.princeton.edu/old/SARC/2.0/pol/"
    csv_files = ["train-balanced.csv.bz2", "test-balanced.csv.bz2"]
    comment_file = "comments.json.bz2"

    # --- Download train/test ---
    for csv_file in csv_files:
        csv_path = os.path.join(save_dir, csv_file)
        extracted_path = csv_path[:-4]

        # 1️⃣ Download
        if not os.path.exists(csv_path):
            print(f"⬇️ Downloading {csv_file} ...")
            urllib.request.urlretrieve(base_url + csv_file, csv_path)
        else:
            print(f"📦 {csv_file} already exists, skipping.")

        # 2️⃣ Extract
        if not os.path.exists(extracted_path):
            print("🔧 Extracting ...")
            with bz2.open(csv_path, "rb") as f_in, open(extracted_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            print(f"✅ Extracted → {extracted_path}")

        # 3️⃣ Add header
        df = pd.read_csv(
            extracted_path,
            sep="|",
            header=None,
            names=["thread_id", "pair", "label"],
            engine="python",
            on_bad_lines="skip",
        )
        out_csv = extracted_path.replace(".csv", "_header.csv")
        df.to_csv(out_csv, index=False)
        print(f"✅ Saved with headers → {out_csv}")

    # --- Download comments.json.bz2 ---
    comments_path = os.path.join(save_dir, comment_file)
    if not os.path.exists(comments_path):
        print(f"⬇️ Downloading {comment_file} ...")
        urllib.request.urlretrieve(base_url + comment_file, comments_path)
        print(f"✅ Downloaded → {comments_path}")
    else:
        print(f"📂 {comment_file} already exists, skipping.")

    print("\n🎯 All done!")
    return [os.path.join(save_dir, f.replace(".bz2", "")) for f in csv_files] + [comments_path]


if __name__ == "__main__":
    paths = download_sarc_balanced_basic()
    print("\n✅ Datasets ready:")
    for p in paths:
        print("   ", p)
