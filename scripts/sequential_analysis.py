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
import argparse
from mindcastlib.pipeline.analysis_pipeline import AnalysisPipeLine
from mindcastlib.configs import AnalysisConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential Analysis Runner")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def find_json_files(root_dir: str, target_name: str = "news_comments.json"):
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f == target_name:
                yield os.path.join(dirpath, f)


def make_output_path(input_path: str, input_root: str, output_root: str) -> str:
    rel_path = os.path.relpath(input_path, input_root)
    rel_dir = os.path.dirname(rel_path)
    out_dir = os.path.join(output_root, rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"infer_{timestamp}.json")


def run_analysis_pipeline(input_json: str, output_json: str):
    try:
        with open(input_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        runner = AnalysisPipeLine(
            analysis_config=AnalysisConfig.SENT_CMT_TOPIC_TTL(),
            realtime=False,
            monitoring=True,
            save=True,
            save_dir=os.path.dirname(output_json),
        )
        runner.run(data)
        print(f"✅ Done: {input_json} → {output_json}")

    except Exception as e:
        print(f"❌ Error: {input_json}: {e}")


def main():
    args = parse_args()
    input_root = args.input_dir
    output_root = args.output_dir

    os.makedirs(output_root, exist_ok=True)

    json_files = list(find_json_files(input_root))
    print(f"🔍 Found {len(json_files)} json files under {input_root}")

    for idx, json_file in enumerate(sorted(json_files)):
        print(f"\n[{idx+1}/{len(json_files)}] Running analysis for: {json_file}")
        out_path = make_output_path(json_file, input_root, output_root)
        run_analysis_pipeline(json_file, out_path)


if __name__ == "__main__":
    main()
