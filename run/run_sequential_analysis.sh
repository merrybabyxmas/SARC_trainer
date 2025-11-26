#!/bin/bash

# ============================================================
# mindcastlib - Sequential Analysis Runner
# 위치: /home/dongwoo38/mindcastlib/run/run_sequential_analysis.sh
# ============================================================

echo "============================================"
echo " 🚀 MindCast Sequential Analysis Pipeline"
echo "============================================"

# ------------------------------------------------------------
# 1. 사용자 설정 영역 (input_dir, output_dir)
# ------------------------------------------------------------
INPUT_DIR="/home/dongwoo38/data/preprocessed_data"
OUTPUT_DIR="/home/dongwoo38/data/analysis_results1"
# 필요하면 위 두 줄만 수정하면 됨.
# ------------------------------------------------------------

# 2. 프로젝트 루트로 이동
cd /home/dongwoo38 || exit
echo "[INFO] Working directory: $(pwd)"

# 3. PYTHONPATH 설정
export PYTHONPATH=/home/dongwoo38:$PYTHONPATH
export INPUT_ROOT="$INPUT_DIR"
export OUTPUT_ROOT="$OUTPUT_DIR"

echo "[INFO] PYTHONPATH : $PYTHONPATH"
echo "[INFO] INPUT_ROOT  : $INPUT_ROOT"
echo "[INFO] OUTPUT_ROOT : $OUTPUT_ROOT"

# 4. 실행 - python -m 방식 우선
echo "[INFO] Running sequential_analysis..."

python -m mindcastlib.scripts.sequential_analysis \
    --input_dir "$INPUT_ROOT" \
    --output_dir "$OUTPUT_ROOT"

# 5. 실패하면 .py 직접 실행
if [ $? -ne 0 ]; then
    echo "[WARN] Module run failed. Retrying with direct script execution..."
    python mindcastlib/scripts/sequential_analysis.py \
        --input_dir "$INPUT_ROOT" \
        --output_dir "$OUTPUT_ROOT"
fi

STATUS=$?

# 6. 실행 결과 처리
if [ $STATUS -eq 0 ]; then
    echo "[SUCCESS] Sequential analysis completed successfully!"
else
    echo "[ERROR] Sequential analysis failed with exit code $STATUS"
fi

echo "============================================"
echo " 🧠 Done."
echo "============================================"
