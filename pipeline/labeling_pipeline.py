from __future__ import annotations
import os 
import json 
import time 
from typing import List, Dict, Tuple, Literal, Optional
from dataclasses import dataclass
import logging
import pprint

import torch

from mindcastlib.configs import LabelingConfig
from mindcastlib.src import apply_func_to_title, apply_func_to_comments, extract_titles, extract_comments, prepare_data_with_temporal_condition, prepare_data
from mindcastlib.src import LLMPipeLine, prompt_and_save_if_missing, load_api_keys


Target = Literal["title", "comments"]
Task = Literal["sentiment", "topic"]


@dataclass
class MCLabelSpec:
    task : Literal["sentiment", "topic"]
    target : Literal["title", "comments"]
    cfg : LabelingConfig
    
    
class ModuleCallable:
    def __init__(self, spec : MCLabelSpec):
        self.task = spec.task 
        self.target = spec.target
        self.cfg = spec.cfg 
        
        device_index = 0 if torch.cuda.is_available() else -1
        
        self.pipe = LLMPipeLine(self.cfg)
        
    def __call__(self, data: Dict) -> Dict:
        if len(data) == 0:
            return {}
        
        if self.target == "title":
            return apply_func_to_title(func = self.pipe, data = data)
        elif self.target == "comments":
            return apply_func_to_comments(func = self.pipe, data = data)
        
        return data
    
    
def build_task_callable(task : Task, target: Target, cfg: LabelingConfig) -> ModuleCallable:
    return ModuleCallable(MCLabelSpec(task = task, target = target, cfg = cfg))




class LabelingPipeLine:
    def __init__(
        self, 
        labeling_config : LabelingConfig | None = None,
        realtime : bool = False,
        monitoring: bool = True,
        save: bool = True,
        save_dir: str | None = None,
    ):
        self.cfg = labeling_config
        self.realtime = realtime
        self.monitoring = monitoring
        self.save = save
        self.save_dir = save_dir or "./outputs"
        os.makedirs(self.save_dir, exist_ok=True)
        
        
        if self.cfg is None:
            logging.info("No config file detected. 기본 설정 파일로 대체 ")
            self.cfg = LabelingConfig.SENT_CMT_TOPIC_TTL() # 댓글 - 감정분류, 제목 - 토픽분류
        
        self._load_models()
        
    def _load_models(self):
        self.runners: Dict[str, ModuleCallable] = {}
        
        if self.cfg.sentiment.active:
            for tgt in self.cfg.sentiment.target:
                key = f"sentiment_{tgt}"
                self.runners[key] = build_task_callable(
                    task = "sentiment", target = tgt, cfg = self.cfg.sentiment.llm
                )
        else:
            logging.info("sentiment excluded!")
            
            
        if self.cfg.topic.active:
            for tgt in self.cfg.topic.target:
                key = f"topic_{tgt}"
                self.runners[key] = build_task_callable(
                    task="topic", target=tgt, cfg=self.cfg.topic.llm
                )
        else:
            logging.info("topic excluded!")
            
            
    def run(self, data:Dict) -> Dict:
        t0 = time.time()
        for key, runner in self.runners.items():
            if self.monitoring:
                logging.info(f"[RUN] executing runner: {key}")
            data = runner(data)
            
        if self.save:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(self.save_dir, f"labeling_{ts}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            if self.monitoring:
                logging.info(f"[Labeling] saved to {out_path}")

        if self.monitoring:
            logging.info(f"[RUN] done in {time.time() - t0:.2f}s")

        return data
    
    
if __name__ == "__main__":
    
    

    
    # ##### 데이터 로딩 ###### 
    # tc = ["2023-05-01", "2023-05-02"]
    # data_dir = "/home/dongwoo38/data/example/ex.json"
    # data = prepare_data_with_temporal_condition(tc, data_dir=data_dir)
    
    
    data_dir = "/home/dongwoo38/data/preprocessed_data/2020/02/01-10/news_comments.json"
    data = prepare_data(data_dir)
    
    
    
    
    ##### config 로딩 ######
    cfg = LabelingConfig.SENT_CMT_TOPIC_TTL() # default : model->"gpt-opensrc", max_token->10 임. 단지 customize가능하다는 점을 보여주기 위해 작성
    model_name = "gpt-opensrc"
    max_token = 10
    cfg.sentiment.llm.llm_name = model_name
    cfg.sentiment.llm.max_token = max_token
    cfg.topic.llm.llm_name = model_name
    cfg.topic.llm.max_token = max_token

    ##### LLM api key 로딩 ######    
    load_api_keys("mindcast.env") #.env 파일 이름을 지정 
    prompt_and_save_if_missing("OPENAI_API_KEY", "OpenAI API key")
    
    
    pipe = LabelingPipeLine(
        labeling_config= cfg,
        realtime=False,
        monitoring=True,
        save=True,
        save_dir="./outputs/labeled"
    )
    
    pipe.run(data)


        
        
        
        

