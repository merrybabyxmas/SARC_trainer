# ============================================================
# ⚙️ config/EVALCONFIG.py
# ============================================================
import torch

EVAL_CFG = {
    # ✅ 기본 디바이스
    "device": "cuda:2" if torch.cuda.is_available() else "cpu",

    # ✅ 평가용 경로
    "paths": {
        "ckpt_path": "/home/dongwoo44/checkpoints_deltaWo_per_layer/best.pt",
        "test_csv": "/home/dongwoo44/data/sarc/test_bal_textlabel.csv",
        "save_dir": "./eval_results",
    },

    # ✅ 데이터셋 관련 설정
    "data": {
        "max_length": 256,
        "batch_size": 32,
        "num_workers": 2,
    },

    # ✅ 기타 설정 (로깅 / 출력 제어 등)
    "misc": {
        "save_predictions": True,
        "save_summary": True,
        "verbose": True,
    }
}
