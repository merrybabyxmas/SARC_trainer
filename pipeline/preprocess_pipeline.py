from mindcastlib.src import preprocess_raw_data
from mindcastlib.configs import PreProcessConfig
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw mental health data.")
    parser.add_argument("--input_dir", type=str, required=True, help="원본 데이터 경로")
    parser.add_argument("--output_dir", type=str, required=True, help="전처리된 데이터 저장 경로")
    parser.add_argument("--save", action="store_true", help="전처리 결과를 저장할지 여부")
    parser.add_argument("--monitoring", action="store_true", help="모니터링 로그 출력 여부")

    args = parser.parse_args()

    res = preprocess_raw_data(
        cfg=PreProcessConfig.DefaultConfig(),
        input_dir=Path(args.input_dir),
        save_dir=Path(args.output_dir),
        save=args.save,
        monitoring=args.monitoring,
    )

    