import os 
from pathlib import Path
import torch 
from mindcastlib.configs import PreProcessConfig
import pprint
from typing import List, Dict, Tuple, Literal
import logging

import json
import re
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import calendar
from tqdm import tqdm

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def group_by_date_range(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    year = date.year
    month = date.month
    day = date.day
    last_day = calendar.monthrange(year, month)[1]

    if 1 <= day <= 10:
        day_range = "01-10"
    elif 11 <= day <= 20:
        day_range = "11-20"
    else:
        day_range = f"21-{last_day:02d}"

    return f"{year}/{month:02d}/{day_range}"

def preprocess_json(cfg: PreProcessConfig, root_dir, base_output_dir=None, save=True, monitoring=False):
    root_dir = Path(root_dir)
    json_files = list(root_dir.rglob("*.json"))

    if monitoring:
        logging.info(f"총 {len(json_files)}개의 JSON 파일을 처리합니다.")
        file_iterator = tqdm(json_files, desc="📄 JSON 파일 처리 중", unit="files")
    else:
        file_iterator = json_files

    illegal_line_separators = re.compile(r"[\u2028\u2029]")
    published_to_posts = defaultdict(list)

    for file_path in file_iterator:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
        except Exception as e:
            logging.warning(f"Failure Loading File: {file_path.name} - {e}")
            continue

        news_date = content[0].get("date", "UNKNOWN")
        title = file_path.stem
        raw_title = title
        comments = []

        for item in content[1:]:
            try:
                comment_info = item["snippet"]["topLevelComment"]["snippet"]
                text = comment_info["textDisplay"]
                published = comment_info["publishedAt"][:10]
                if text.strip():
                    clean_text = illegal_line_separators.sub(" ", text.replace("\n", " ").replace("<br>", " ")).strip()
                    comments.append((published, clean_text))
            except KeyError:
                continue

        published_group = defaultdict(list)
        for published, comment in comments:
            published_group[published].append(comment)

        for published_date, grouped_comments in published_group.items():
            if not grouped_comments:
                continue
            all_values = {
                "title": title,
                "raw_title": raw_title,
                "news_date": news_date,
                "comments": grouped_comments
            } # TODO :[정선] all values 추가 (추후)

            post = {k: v for k, v in all_values.items() if k in cfg.included}


            published_to_posts[published_date].append(post)

    if monitoring:
        logging.info("💾 저장 중...")
        date_iterator = tqdm(sorted(published_to_posts.keys()), desc="📁 날짜별 저장", unit="dates")
    else:
        date_iterator = sorted(published_to_posts.keys())

    base_output_dir = Path(base_output_dir) if base_output_dir else root_dir.parent / "preprocessed_data"
    output_results = []

    for pub_date in date_iterator:
        posts = published_to_posts[pub_date]
        posts.sort(key=lambda x: x.get("news_date", ""))

        output_data = {
            "data": [
                {
                    "date": pub_date,
                    "posts": posts
                }
            ]
        }

        subfolder = group_by_date_range(pub_date)
        save_dir = base_output_dir / subfolder
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / "news_comments.json"

        try:
            if save:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=4)
                output_results.append(output_data)
        except Exception as e:
            logging.error(f"[Fail] {output_file} - {e}")

    if monitoring:
        print(f"모든 작업이 완료되었습니다. \n{base_output_dir}")

    return output_results




def preprocess_raw_data(cfg:PreProcessConfig, 
                        input_dir: Path | str = None, 
                        save_dir:Path | str = None,
                        save = True,
                        monitoring = True) -> None:
    r"""
    save : 전처리한 데이터를 저장할 것인지 판단. 기본값은 저장하지만 저장공간 혹은 실시간이 필요할 때는 off도 구현 
    monitoring : 전처리 하는데에 소요된 시간 및 저장용량등을 tracking 
    """
    
    ### 1. input_dir에서 데이터를 불러와서
    ### 2. 전처리 후  
    ### 3. save 에 따라 저장 

    if not cfg.included:
        logging.warning("No included fields specified. Defaulting to ['title', 'raw_title', ...]")
        cfg = PreProcessConfig.DefaultConfig()
    
    if save is False:
        logging.info("실시간 모드를 위해 save를 껐습니다.")
        
    if input_dir is None:
        logging.info("no raw data input directory detected! put input_dir")
    
    if save_dir is None and save is True:
        logging.info("No save dir detected. put save dir")

    return preprocess_json(
        cfg = cfg,
        root_dir=input_dir,
        base_output_dir=save_dir,
        save=save,
        monitoring=monitoring
    )



# TODO: [대원]
class GrammerChecker:
    def __init__(self,
                 method = Literal["re", "snu"]):
        pass
    def forward(self, data:List[str]) -> List[str]:
        # 데이터 : List[str] -> List[str] 형식 꼭 맞춰서 만들어줄 것.
        # ex. data : ["안녕하쉐요", 오늘참날 씨가덥 다" ] -> GrammerChecker -> ["안녕하세요", "오늘 참 날씨가 덥다"]
        # 이렇게 결과가 나오도록 !  
        pass






# 예시 실행 
# python -m mindcastlib.src.preprocess_utils

if __name__ == "__main__":
    cfg = PreProcessConfig.DefaultConfig()
    pprint.pprint(cfg.model_dump_json())
    
    input_dir = Path("/home/jeongseon38/MLLAB/mentalHealth/data/original_data")          # data 경로로 수정 필요
    save_dir = Path("/home/jeongseon38/MLLAB/mentalHealth/data/preprocessed_data")       # data 경로로 수정 필요
    save=True
    monitoring=True

    res = preprocess_raw_data(
        cfg=cfg,
        input_dir=input_dir,
        save_dir=save_dir,
        save=True,
        monitoring=True
    )    

    # pprint.pprint(res)