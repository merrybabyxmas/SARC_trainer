#!/bin/bash

# ============================================================
# MindCast — Single JSON Analysis Runner
# 위치: /home/dongwoo38/mindcastlib/run/run_single_analysis.sh
# ============================================================

echo "============================================"
echo " 🚀 MindCast Single File Analysis"
echo "============================================"

# ------------------------------------------------------------
# 1. 사용자 설정 영역 (필요시 직접 수정)
# ------------------------------------------------------------
INPUT_JSON="/home/dongwoo38/data/preprocessed_data/2020/02/01-10/news_comments.json"
OUTPUT_DIR="/home/dongwoo38/mindcastlib/outputs/single_analysis2"
CONFIG_NAME="SENT_CMT_TOPIC_TTL"   # 기본 config 유지
# ------------------------------------------------------------

mkdir -p "$OUTPUT_DIR"

# 2. 프로젝트 루트 이동
cd /home/dongwoo38 || exit
echo "[INFO] Working directory: $(pwd)"

# 3. PYTHONPATH 등록
export PYTHONPATH=/home/dongwoo38:$PYTHONPATH
echo "[INFO] PYTHONPATH : $PYTHONPATH"

# 4. AnalysisPipeline 단일 파일 실행
echo "[INFO] Running AnalysisPipeLine..."
python - << EOF
from mindcastlib.configs import AnalysisConfig
from mindcastlib.pipeline.analysis_pipeline import AnalysisPipeLine
from mindcastlib.src import prepare_data
import pprint

input_path = "$INPUT_JSON"
save_dir = "$OUTPUT_DIR"

print(f"[INFO] Loading file: {input_path}")
data = prepare_data(input_path)

runner = AnalysisPipeLine(
    analysis_config=getattr(AnalysisConfig, "$CONFIG_NAME")(),
    realtime=False,
    monitoring=True,
    save=True,
    save_dir=save_dir
)

result = runner.run(data)
print("[INFO] Analysis Complete!")
pprint.pprint(result)
EOF

STATUS=$?

# 5. 종료 메시지
if [ $STATUS -eq 0 ]; then
    echo "[SUCCESS] Single-file analysis finished!"
else
    echo "[ERROR] Analysis failed with exit code $STATUS"
fi

echo "============================================"
echo " 🧠 Done."
echo "============================================"
