from datasets import load_dataset
import pandas as pd
import os

def download_kor_sarc(save_dir="./data/sarc"):
    os.makedirs(save_dir, exist_ok=True)
    print("⬇️ Downloading KorSarcasmClassification from Hugging Face...")

    dataset = load_dataset("mteb/KorSarcasmClassification")

    for split in ["train", "test"]:
        if split not in dataset:
            continue
        df = pd.DataFrame(dataset[split])
        df = df.rename(columns={"text": "text", "label": "label"})
        out_path = os.path.join(save_dir, f"korsarc_{split}.csv")
        df.to_csv(out_path, index=False)
        print(f"✅ Saved {split} → {out_path} ({len(df)} samples)")

    print("🎯 All KorSarcasmClassification splits downloaded!")

if __name__ == "__main__":
    download_kor_sarc()
