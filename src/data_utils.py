# from mindcastlib.configs import SENT_CMT_TOPIC_TTL_CONFIG
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Dict, Tuple, Callable, Any, Iterable
import json
import logging
import pprint

import torch


def prepare_data(
    data_dir: str = None   
    ) -> Dict:
    if data_dir is None:
        raise FileNotFoundError(f"you should put data directory address, current : {data_dir}")
    else:
        logging.info(f"Load data from {data_dir}")
        with open(data_dir, "r") as f:
            raw_data = json.load(f)
            return raw_data
    return {}

        

def prepare_data_with_temporal_condition(
    tc: List[str] | None,
    data : Dict | None = None,
    data_dir: str = None   
    ) -> Dict:
    """
    원하는 time condition의 데이터를 추출 
    
    """
    if tc is None:
        logging.info("Real time Mode : when TC is None")
        return {}  # TODO: 실시간 모드 구현
    else:
        if data is None and data_dir is not None:
            logging.info(f"Load data from {data_dir}")
            with open(data_dir, "r") as f:
                raw_data = json.load(f)
                return filter_data_with_temporal_condition(raw_data, tc)
        elif data is not None and data_dir is None:
            logging.info(f"use given data. not from data directory")
            return filter_data_with_temporal_condition(data, tc)
        elif data is not None and data_dir is not None:
            raise ValueError("데이터와 데이터디렉토리 둘 중 한개만 주어져야 합니다.")
        else:
            raise ValueError("데이터와 데이터 디렉토리 둘 중 최소 한개는 주어져야 합니다. ")
            



def filter_data_with_temporal_condition(data: Dict, tc:List[str]) -> Dict:
    r"데이터가 dict 형식으로 들어왔을 때, tc에 맞춰서  dict 반환"
    start_date, end_date = tc[0], tc[-1]
    return {
        "data": [
            entry for entry in data.get("data", [])
            if start_date <= entry.get("date", "") <= end_date
        ]
        }
            
            
def extract_titles(data: Dict) -> List[str]:
    titles: List[str] = []
    for day in data.get("data", []):
        for post in day.get("posts", []):
            t = post.get("title")
            if t is not None:
                titles.append(t)
    return titles

# 2) 모든 comments를 평탄화해서 리스트로 추출
def extract_comments(data: Dict) -> List[str]:
    comments: List[str] = []
    for day in data.get("data", []):
        for post in day.get("posts", []):
            for c in post.get("comments", []):
                if c is not None:
                    comments.append(c)
    return comments



def extract_analysis_from_title(data: Dict, key: str, labels_only: bool = False) -> List[Any]:
    """
    post.analyses[key] 안의 결과를 리스트로 추출.
    labels_only=True면 label 문자열만 뽑아서 반환.
    """
    results: List[Any] = []
    for day in data.get("data", []):
        for post in day.get("posts", []):
            analyses = post.get("analyses", {})
            if key in analyses:
                val = analyses[key]
                if labels_only:
                    if isinstance(val, dict) and "label" in val:
                        results.append(val["label"])
                    elif isinstance(val, dict):
                        results.append(val["label"])
                    
                else:
                    results.append(val)
    return results


def extract_analysis_from_comments(data: Dict, key: str, labels_only: bool = False) -> List[Any]:
    """
    post.analyses[key] 안의 결과를 평탄화해서 반환.
    labels_only=True면 label 문자열만 뽑아서 리스트로 반환.
    """
    results: List[Any] = []
    # print(f"data : {data}")
    # print(f"key : {key}")
    
    for day in data.get("data", []):
        for post in day.get("posts", []):
            analyses = post.get("analyses", {})
            if key in analyses:
                for r in analyses[key]:  # r은 [{'label':..,'score':..}] 꼴
                    print(r)
                    if labels_only:
                        # 가장 높은 score의 label만 가져오려면:
                        if isinstance(r, list) and len(r) > 0:
                            results.append(r[0]["label"])
                        elif isinstance(r, dict):
                            results.append(r["label"])
                            
                    else:
                        results.append(r)
    return results


def apply_func_to_title(
    func: Callable | None = None,
    func_name: str | None = None,
    data: Dict | None = None,
    batch_size: int = 32,
    **kwargs
) -> Dict:
    if func is None:
        raise ValueError("you should put Callable model e.g. Sentiment classification Model...")
    if data is None:
        raise ValueError("You should put data!")

    func_name = func_name if func_name is not None else func.__class__.__name__

    # ----------------------------------------------------
    # 1️⃣ 모든 title 수집
    # ----------------------------------------------------
    all_titles: List[str] = []
    for day in data.get("data", []):
        for post in day.get("posts", []):
            if post.get("title"):
                all_titles.append(post["title"])

    # ----------------------------------------------------
    # 2️⃣ 배치 inference
    # ----------------------------------------------------
    results = []
    for i in range(0, len(all_titles), batch_size):
        batch = all_titles[i:i + batch_size]
        batch_results = func(batch, **kwargs)
        results.extend(batch_results)

    func_results_iter = iter(results)

    # ----------------------------------------------------
    # 3️⃣ 결과 저장
    # ----------------------------------------------------
    for day in data.get("data", []):
        for post in day.get("posts", []):
            title = post.get("title")
            if title is None:
                continue

            r = next(func_results_iter)


            if func_name == "SuicideDetectionPipeLine":
                # r: {"suicide_related":..., "suicide_keyword_mask":..., "suicide_subtag_mask":...}
                res = {
                    "suicide_related": r["suicide_related"],
                    "suicide_keyword_mask": r["suicide_keyword_mask"],
                    "suicide_subtag_mask": r["suicide_subtag_mask"],
                }

            # ------------------------------
            # HF 모델: 기존 로직 유지
            # ------------------------------
            elif isinstance(r, str):
                res = {"label": r}
            else:
                res = r

            analyses = post.setdefault("analyses", {})
            analyses[func_name + "_title"] = [res]

    return data



def apply_func_to_comments(
    func: Callable | None = None,
    func_name: str | None = None,
    data: Dict | None = None,
    batch_size: int = 32,      # ✅ 추가
    **kwargs
) -> Dict:
    if func is None:
        raise ValueError("you should put Callable model e.g. Sentiment classification Model...")
    if data is None:
        raise ValueError("You should put data!")

    func_name = func_name if func_name is not None else func.__class__.__name__

    # --------------------------------------------
    # 1️⃣ 전체 댓글 수집
    # --------------------------------------------
    all_comments: List[str] = []
    for day in data.get("data", []):
        for post in day.get("posts", []):
            for comment in post.get("comments", []):
                all_comments.append(comment)

    # --------------------------------------------
    # 2️⃣ 배치 단위 inference 수행
    # --------------------------------------------
    results = []
    for i in range(0, len(all_comments), batch_size):
        batch = all_comments[i:i + batch_size]
        batch_results = func(batch, **kwargs)
        results.extend(batch_results)

    func_results_iter = iter(results)

    # --------------------------------------------
    # 3️⃣ 결과를 다시 데이터 구조에 삽입
    # --------------------------------------------
    for day in data.get("data", []):
        for post in day.get("posts", []):
            comments = post.get("comments", [])
            n = len(comments)
            if n == 0:
                continue

            per_post_results = []
            for _ in range(n):
                r = next(func_results_iter)
                if isinstance(r, str):
                    per_post_results.append([{"label": r}])
                elif isinstance(r, dict) and "label" in r:
                    per_post_results.append([r])
                else:
                    per_post_results.append(r)

            analyses = post.setdefault("analyses", {})
            analyses[func_name + "_comments"] = per_post_results

    return data

def apply_func_to_something_from_titlelike_double_data(
    func: Callable,
    target1: str,
    target2: str,
    data1: Dict,
    data2: Dict,
    func_name: str | None = None,
    labels_only: bool = False,
    **kwargs
) -> Dict:
    true_vals = extract_analysis_from_title(data1, target1, labels_only=labels_only)
    pred_vals = extract_analysis_from_title(data2, target2, labels_only=labels_only)
    return func(true_vals, pred_vals, **kwargs)


def apply_func_to_something_from_commentlike_double_data(
    func: Callable,
    target1: str,
    target2: str,
    data1: Dict,
    data2: Dict,
    func_name: str | None = None,
    labels_only: bool = False,
    **kwargs
) -> Dict:
    print(f"true comments:")
    true_vals = extract_analysis_from_comments(data1, target1, labels_only=labels_only)
    print(f"pred comments:")
    pred_vals = extract_analysis_from_comments(data2, target2, labels_only=labels_only)
    return func(true_vals, pred_vals, **kwargs)

    

    
if __name__ == "__main__":
    # config import 테스트
    # config = SENT_CMT_TOPIC_TTL_CONFIG()
    # print(config.model_dump_json(indent = 2))
    
    # from mindcastlib.configs import DefaultModuleConfig
    # cfg = DefaultModuleConfig()
    # print(cfg.sentiment_model.macro_labels)
    
    
    # temporal conditon 테스트 
    tc = ["2023-05-01", "2023-05-02"]
    data_dir = "/home/dongwoo38/data/example/ex.json"
    res = prepare_data_with_temporal_condition(tc, data_dir=data_dir)
    # pprint.pprint(res)
    
    # print(extract_titles(res))
    # print(extract_comments(res))
    
    
    # sent_func = HFSentimentFunc(
    #     model_name="hun3359/klue-bert-base-sentiment",
    #     device=0 if torch.cuda.is_available() else -1,
    #     batch_size=32,
    #     return_all_scores=False,  # True로 하면 모든 라벨 점수 반환
    #     max_length=256
    # )

    # # 3) 타이틀/댓글에 적용
    # res = _apply_func_to_title(func=sent_func, data=res, func_name="hf_sent")
    # res = _apply_func_to_comments(func=sent_func, data=res, func_name="hf_sent")
    # pprint.pprint(res)



    # extract analysis
    tc = ["2023-05-01", "2023-05-02"]
    data_dir = "/home/dongwoo38/outputs/analysis/infer_20250820_133357.json"
    title_target = "TextClassificationPipeline_title" 
    comment_target = "SentimentClassification_comments"
    res = prepare_data_with_temporal_condition(tc, data_dir=data_dir)
    print(extract_analysis_from_comments(res, comment_target, labels_only= True))
    print(extract_analysis_from_title(res, title_target, labels_only=True))    
        
    
    