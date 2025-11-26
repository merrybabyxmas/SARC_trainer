from __future__ import annotations
import os
import json
import time
from typing import List, Dict, Tuple, Literal, Optional, Union
from dataclasses import dataclass 
import logging
import pprint
import torch
from transformers import pipeline

from mindcastlib.configs import BaseConfig, AnalysisConfig, AnalysisUnit
from mindcastlib.src import (
    apply_func_to_title, apply_func_to_comments,
    extract_titles, extract_comments,
    prepare_data_with_temporal_condition, prepare_data,
    should_collapse6, macro_slices_60x6, predict_6sentiments
)
from mindcastlib.src.sarc_utils import load_sarcasm_model, predict_sarcasm

Target = Literal["title", "comments"]
Task = Literal["sentiment", "topic", "summary", "sarcasm"]


# ============================================================
# 🧩 각 Task별 디바이스 설정
# ============================================================
# sentiment/topic/classifier → cuda:0
# sarcasm → cuda:1
DEVICE_MAP = {
    "sentiment": "cuda:0" if torch.cuda.device_count() > 0 else "cpu",
    "topic": "cuda:0" if torch.cuda.device_count() > 0 else "cpu",
    "classifier": "cuda:0" if torch.cuda.device_count() > 0 else "cpu",
    "sarcasm": "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
}


@dataclass
class MCPipeSpec:
    task: Literal["sentiment", "topic", "summary", "sarcasm"]
    target: Literal["title", "comments"]
    cfg: BaseConfig


class ModuleCallable:
    HF_TASK_MAPING: Dict[str, str] = {
        "sentiment": "text-classification",
        "topic": "text-classification",
        "summary": "summarization",
        "sarcasm": None,
    }

    def __init__(self, spec: MCPipeSpec):
        self.task = spec.task
        self.target = spec.target
        self.cfg = spec.cfg
        self.device = DEVICE_MAP.get(self.task, "cpu")

        if self.task == "sarcasm":
            print(f"[INIT] 🧠 Loading sarcasm model on {self.device}")
            self.model, self.tokenizer = load_sarcasm_model(
                device=self.device
            )
            self.pipe = None
        else:
            device_index = int(self.device.split(":")[-1]) if "cuda" in self.device else -1
            print(f"[INIT] ⚙️ Loading {self.task} pipeline on {self.device}")
            self.pipe = pipeline(
                self.HF_TASK_MAPING[self.task],
                model=self.cfg.model_name,
                device=device_index,
            )

        self._collapse6 = False
        if self.task == "sentiment":
            self._macro_labels = getattr(self.cfg, "macro_labels", None)
            self._macro_slices = macro_slices_60x6()
            self._collapse6 = should_collapse6(self.task, self.pipe.model)

    def __call__(self, data: Dict) -> Dict:
        if len(data) == 0:
            return {}

        if self.task == "sarcasm":
            def func(texts):
                return predict_sarcasm(
                    texts=texts,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    max_length=self.cfg.max_length,
                )
            fn = "SarcasmDetectionPipeLine"

        elif self._collapse6:
            def func(texts):
                return predict_6sentiments(
                    texts=texts,
                    pipe=self.pipe,
                    macro_slices=self._macro_slices,
                    **self.cfg.model_dump()
                )
            fn = "SentimentClassificationPipeLine"
        else:
            func = self.pipe
            fn = self.pipe.task

        if self.target == "title":
            return apply_func_to_title(func=func, func_name=fn, data=data)
        elif self.target == "comments":
            return apply_func_to_comments(func=func, func_name=fn, data=data)
        return data


def build_task_callable(task: Task, target: Target, cfg: BaseConfig) -> ModuleCallable:
    return ModuleCallable(MCPipeSpec(task=task, target=target, cfg=cfg))


class AnalysisPipeLine:
    def __init__(
        self,
        analysis_config: AnalysisConfig | None = None,
        realtime: bool = False,
        monitoring: bool = True,
        save: bool = True,
        save_dir: str | None = None,
    ):
        self.cfg = analysis_config
        self.realtime = realtime
        self.monitoring = monitoring
        self.save = save
        self.save_dir = save_dir or "./outputs"
        os.makedirs(self.save_dir, exist_ok=True)

        if self.cfg is None:
            logging.info("No config file detected. 기본 설정 파일로 대체 ")
            self.cfg = AnalysisConfig.SENT_CMT_TOPIC_TTL()

        self._load_models()

    def _load_models(self):
        self.runners: Dict[str, ModuleCallable] = {}

        if self.cfg.sarcasm.active:
            for tgt in self.cfg.sarcasm.target:
                key = f"sarcasm_{tgt}"
                self.runners[key] = build_task_callable(
                    task="sarcasm", target=tgt, cfg=self.cfg.sarcasm.module
                )
        else:
            logging.info("sarcasm excluded!")

        if self.cfg.sentiment.active:
            for tgt in self.cfg.sentiment.target:
                key = f"sentiment_{tgt}"
                self.runners[key] = build_task_callable(
                    task="sentiment", target=tgt, cfg=self.cfg.sentiment.module
                )
        else:
            logging.info("sentiment excluded!")

        if self.cfg.topic.active:
            for tgt in self.cfg.topic.target:
                key = f"topic_{tgt}"
                self.runners[key] = build_task_callable(
                    task="topic", target=tgt, cfg=self.cfg.topic.module
                )
        else:
            logging.info("topic excluded!")

        if self.cfg.classifier.active:
            for tgt in self.cfg.classifier.target:
                key = f"classifier_{tgt}"
                self.runners[key] = build_task_callable(
                    task="sentiment",
                    target=tgt,
                    cfg=self.cfg.classifier.module,
                )
        else:
            logging.info("classifier excluded!")

    def run(self, data: Dict) -> Dict:
        t0 = time.time()
        for key, runner in self.runners.items():
            if self.monitoring:
                logging.info(f"[RUN] executing runner: {key}")
                print(f" → Using device: {runner.device}")
            data = runner(data)

        if self.save:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(self.save_dir, f"infer_{ts}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            if self.monitoring:
                logging.info(f"[Inference] saved to {out_path}")

        if self.monitoring:
            logging.info(f"[RUN] done in {time.time() - t0:.2f}s")

        return data


if __name__ == "__main__":
    data_dir = "/home/dongwoo38/data/preprocessed_data/2020/02/01-10/news_comments.json"
    data = prepare_data(data_dir)

    runner = AnalysisPipeLine(
        analysis_config=AnalysisConfig.SENT_CMT_TOPIC_TTL(),
        realtime=False,
        monitoring=True,
        save=True,
        save_dir="./outputs/ex"
    )

    result = runner.run(data)
    pprint.pprint(result)
