# ============================================================
# 🧭 utils/logger.py
# ============================================================

import logging, os, matplotlib
import matplotlib.font_manager as fm

def setup_logger(name="DeltaWoTrainer"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

logger = setup_logger()

# ✅ Optional: 한글 폰트 자동 설정
font_path = os.path.expanduser("~/.local/share/fonts/NanumGothic.ttf")
if os.path.exists(font_path):
    font_name = fm.FontProperties(fname=font_path).get_name()
    matplotlib.rcParams["font.family"] = [font_name]
matplotlib.rcParams["axes.unicode_minus"] = False
