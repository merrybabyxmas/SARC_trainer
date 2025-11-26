# ============================================================
# 📦 scripts/sequential_analysis.py
#  - preprocessed_data 전체를 순회하며 news_comments.json 분석
#  - 결과를 동일한 구조로 analysis_results 폴더에 저장
# ============================================================

from __future__ import annotations
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import pprint

from mindcastlib.pipeline.analysis_pipeline import AnalysisPipeLine
from mindcastlib.configs import AnalysisConfig
from mindcastlib.src import prepare_data


# ------------------------------------------------------------
# 설정
# ------------------------------------------------------------
INPUT_ROOT = "/home/yein40/data/2020"
OUTPUT_ROOT = "/home/yein40/data/results"

# 로그 설정
os.makedirs(OUTPUT_ROOT, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(OUTPUT_ROOT, "sequential_analysis.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def find_json_files(root_dir: str, target_name: str = "news_comments.json"):
    """
    root_dir 하위에서 target_name을 가진 모든 파일의 경로를 재귀적으로 찾는다.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f == target_name:
                yield os.path.join(dirpath, f)


def make_output_path(input_path: str) -> str:
    """
    입력 파일 경로를 기반으로 동일한 디렉토리 구조로 출력 디렉토리를 만든다.
    예: preprocessed_data/2020/01/01-10/news_comments.json →
        analysis_results/2020/01/01-10/infer_20251108_001530.json
    """
    rel_path = os.path.relpath(input_path, INPUT_ROOT)
    rel_dir = os.path.dirname(rel_path)
    out_dir = os.path.join(OUTPUT_ROOT, rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"infer_{timestamp}.json")
    return out_path

def run_analysis_pipeline(input_json: str, output_json: str):
    """
    하나의 json 파일에 대해 AnalysisPipeLine 실행
    """
    logging.info(f"[RUN] Processing file: {input_json}")
    try:
        # ✅ UTF-8로 강제 로드 (prepare_data 대신 직접 json.load)
        with open(input_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        runner = AnalysisPipeLine(
            analysis_config=AnalysisConfig.SENT_CMT_TOPIC_TTL(),
            realtime=False,
            monitoring=True,
            save=True,
            save_dir=os.path.dirname(output_json),
        )
        result = runner.run(data)
        logging.info(f"[DONE] Saved result to {output_json}")
        print(f"✅ Done: {input_json} → {output_json}")

    except UnicodeDecodeError:
        # ✅ 일부 파일이 UTF-8이 아닌 경우 대비 (예: euc-kr)
        try:
            with open(input_json, "r", encoding="euc-kr") as f:
                data = json.load(f)
            runner = AnalysisPipeLine(
                analysis_config=AnalysisConfig.SENT_CMT_TOPIC_TTL(),
                realtime=False,
                monitoring=True,
                save=True,
                save_dir=os.path.dirname(output_json),
            )
            result = runner.run(data)
            logging.info(f"[DONE] Saved result to {output_json}")
            print(f"✅ Done (euc-kr): {input_json} → {output_json}")
        except Exception as e:
            logging.error(f"[ERROR][ENCODING_FAIL] {input_json}: {e}")
            print(f"❌ Encoding error for {input_json}: {e}")

    except Exception as e:
        logging.error(f"[ERROR] {input_json}: {e}")
        print(f"❌ Error processing {input_json}: {e}")




def main():
    json_files = list(find_json_files(INPUT_ROOT))
    print(f"🔍 Found {len(json_files)} json files under {INPUT_ROOT}")

    for idx, json_file in enumerate(sorted(json_files)):
        print(f"\n[{idx+1}/{len(json_files)}] Running analysis for: {json_file}")
        out_path = make_output_path(json_file)
        run_analysis_pipeline(json_file, out_path)


if __name__ == "__main__":
    main()
