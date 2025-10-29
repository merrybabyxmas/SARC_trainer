import pandas as pd

# 원본 파일 (headered version)
train_path = "/home/dongwoo44/data/sarc/train-balanced.csv"
test_path = "/home/dongwoo44/data/sarc/test-balanced.csv"

# 변환 후 저장할 경로
train_out = "/home/dongwoo44/data/sarc/train_bal_textlabel.csv"
test_out = "/home/dongwoo44/data/sarc/test_bal_textlabel.csv"

# CSV 로드
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 원본 라벨이 '1 0' 또는 '0 1' 형태이므로, 첫 번째 숫자만 추출
train_df["label"] = train_df["label"].astype(str).str.split().str[0].astype(int)
test_df["label"] = test_df["label"].astype(str).str.split().str[0].astype(int)

# 여기서는 그냥 'pair'를 모델 입력 text로 사용 (나중에 댓글 본문 매핑 가능)
train_df["text"] = train_df["pair"]
test_df["text"] = test_df["pair"]

# 필요한 컬럼만 남기고 저장
train_df[["text", "label"]].to_csv(train_out, index=False)
test_df[["text", "label"]].to_csv(test_out, index=False)

print(f"✅ train -> {train_out}")
print(f"✅ test  -> {test_out}")
