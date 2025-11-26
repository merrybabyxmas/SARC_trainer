#!/bin/bash

# ============================================
# mindcastlib - Preprocess Pipeline Runner
# 위치: /home/dongwoo38/mindcastlib/run/run_preprocess.sh
# ============================================

# 1. 프로젝트 루트로 이동
cd /home/dongwoo38

echo "[INFO] Working directory: $(pwd)"

# 2. PYTHONPATH 환경 변수 설정
export PYTHONPATH=/home/dongwoo38:$PYTHONPATH
echo "[INFO] PYTHONPATH set to $PYTHONPATH"

# 3. 경로 설정
INPUT_DIR="/home/dongwoo38/data/original_data"
OUTPUT_DIR="/home/dongwoo38/data/preprocessed_data1"

echo "[INFO] Input Directory : $INPUT_DIR"
echo "[INFO] Output Directory: $OUTPUT_DIR"

# 4. 파이프라인 실행
echo "[INFO] Running preprocess pipeline..."

python mindcastlib/pipeline/preprocess_pipeline.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --save \
    --monitoring

STATUS=$?

# 5. 실행 결과 처리
if [ $STATUS -eq 0 ]; then
    echo "[SUCCESS] Preprocessing completed successfully!"
else
    echo "[ERROR] Preprocessing failed with exit code $STATUS"
fi
