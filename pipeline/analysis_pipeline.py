from __future__ import annotations
import os
import json
import time
from typing import List, Dict, Literal
from dataclasses import dataclass
import logging
import pprint

import torch
from transformers import pipeline

from mindcastlib.configs import BaseConfig, AnalysisConfig
from mindcastlib.src import (
    apply_func_to_title, apply_func_to_comments,
    extract_titles, extract_comments,
    prepare_data_with_temporal_condition, prepare_data,
    should_collapse6, macro_slices_60x6, predict_6sentiments,
)
from mindcastlib.src.sarc_utils import load_sarcasm_model, predict_sarcasm

Target = Literal["title", "comments"]
Task = Literal["sentiment", "topic", "summary", "sarcasm", "suicide"]


# ============================================================
# 🧩 각 Task별 디바이스 설정
#   - sentiment/topic/classifier → cuda:0
#   - sarcasm → cuda:1 (있으면), 없으면 cuda:0
#   - suicide → cuda:0 (KoSimCSE)
# ============================================================
DEVICE_MAP = {
    "sentiment": "cuda:0" if torch.cuda.device_count() > 0 else "cpu",
    "topic": "cuda:0" if torch.cuda.device_count() > 0 else "cpu",
    "classifier": "cuda:0" if torch.cuda.device_count() > 0 else "cpu",
    "sarcasm": "cuda:1" if torch.cuda.device_count() > 1 else (
        "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
    ),
    "suicide": "cuda:0" if torch.cuda.device_count() > 0 else "cpu",
}


# ============================================================
# 🔧 MCPipeSpec — task/target/config 묶음
# ============================================================
@dataclass
class MCPipeSpec:
    task: Task
    target: Target
    cfg: BaseConfig


# ============================================================
# 🔧 ModuleCallable — 각 task별 호출 래퍼
# ============================================================
class ModuleCallable:
    HF_TASK_MAPING: Dict[str, str] = {
        "sentiment": "text-classification",
        "topic": "text-classification",
        "summary": "summarization",
        "sarcasm": None,   # HF pipeline 안 씀
        # suicide 도 HF pipeline 안 씀
    }

    def __init__(self, spec: MCPipeSpec):
        self.task = spec.task
        self.target = spec.target
        self.cfg = spec.cfg
        self.device = DEVICE_MAP.get(self.task, "cpu")

        # ----- Sarcasm (커스텀 모델) -----
        if self.task == "sarcasm":
            print(f"[INIT] 🧠 Loading sarcasm model on {self.device}")
            self.model, self.tokenizer = load_sarcasm_model(
                device=self.device
            )
            self.pipe = None

        # ----- Suicide (SimilaritySearchModel) -----
        elif self.task == "suicide":
            from mindcastlib.src.suicide_utils import SimilaritySearchModel

            print(f"[INIT] 💀 Loading suicide SimilaritySearchModel on {self.device}")

            # 🔥 현재 설치된 mindcastlib 패키지의 루트 자동 탐지
            import mindcastlib
            pkg_root = os.path.dirname(mindcastlib.__file__)
            suicide_cfg_dir = os.path.join(pkg_root, "configs", "suicide")

            # cfg(dict)로 변환 후 suicide_config_root 추가
            suicide_dict = self.cfg.model_dump()
            suicide_dict["suicide_config_root"] = suicide_cfg_dir

            # 실행
            self.model = SimilaritySearchModel(suicide_dict)
            self.pipe = None


        # ----- HF pipeline 계열 (sentiment/topic/summary/classifier) -----
        else:
            device_index = int(self.device.split(":")[-1]) if "cuda" in self.device else -1
            print(f"[INIT] ⚙️ Loading {self.task} pipeline on {self.device}")
            self.pipe = pipeline(
                self.HF_TASK_MAPING[self.task],
                model=self.cfg.model_name,
                device=device_index,
            )

        # ----- sentiment 전용: 60-way → 6-way collapse 여부 -----
        self._collapse6 = False
        if self.task == "sentiment":
            self._macro_labels = getattr(self.cfg, "macro_labels", None)
            self._macro_slices = macro_slices_60x6()
            self._collapse6 = should_collapse6(self.task, self.pipe.model)

    # --------------------------------------------------------
    # 🧩 실행 (data: Dict → Dict)
    # --------------------------------------------------------
    def __call__(self, data: Dict) -> Dict:
        if len(data) == 0:
            return {}

        # ----- Sarcasm -----
        if self.task == "sarcasm":
            def func(texts: List[str]):
                return predict_sarcasm(
                    texts=texts,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    max_length=self.cfg.max_length,
                )
            fn = "SarcasmDetectionPipeLine"

        # ----- Suicide -----
        elif self.task == "suicide":
            def func(texts: List[str]):
                """
                SimilaritySearchModel(titles) → List[Dict]
                Dict 내부:
                  - suicide_related: bool
                  - keyword_mask: Dict[str, bool]
                  - subtag_mask: Dict[str, bool]
                """
                results = self.model(texts)
                out = []
                for r in results:
                    out.append({
                        "suicide_related": r["suicide_related"],
                        "suicide_keyword_mask": r["keyword_mask"],
                        "suicide_subtag_mask": r["subtag_mask"],
                    })
                return out
            fn = "SuicideDetectionPipeLine"

        # ----- Sentiment(특수: 60-way → 6-way collapse) -----
        elif self._collapse6:
            def func(texts: List[str]):
                return predict_6sentiments(
                    texts=texts,
                    pipe=self.pipe,
                    macro_slices=self._macro_slices,
                    **self.cfg.model_dump()
                )
            fn = "SentimentClassificationPipeLine"

        # ----- 일반 HF pipeline (topic/summary/기타 sentiment) -----
        else:
            func = self.pipe
            fn = self.pipe.task

        # ----- title / comments 에 적용 -----
        if self.target == "title":
            return apply_func_to_title(func=func, func_name=fn, data=data)
        elif self.target == "comments":
            return apply_func_to_comments(func=func, func_name=fn, data=data)
        return data


# ============================================================
# 🔧 빌더
# ============================================================
def build_task_callable(task: Task, target: Target, cfg: BaseConfig) -> ModuleCallable:
    return ModuleCallable(MCPipeSpec(task=task, target=target, cfg=cfg))


# ============================================================
# 🧩 메인 AnalysisPipeLine
# ============================================================
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

    # --------------------------------------------------------
    # 🧩 Runner 모듈 로딩
    # --------------------------------------------------------
    def _load_models(self):
        self.runners: Dict[str, ModuleCallable] = {}

        # ----- sarcasm -----
        if self.cfg.sarcasm.active:
            for tgt in self.cfg.sarcasm.target:
                key = f"sarcasm_{tgt}"
                self.runners[key] = build_task_callable(
                    task="sarcasm", target=tgt, cfg=self.cfg.sarcasm.module
                )
        else:
            logging.info("sarcasm excluded!")

        # ----- sentiment -----
        if self.cfg.sentiment.active:
            for tgt in self.cfg.sentiment.target:
                key = f"sentiment_{tgt}"
                self.runners[key] = build_task_callable(
                    task="sentiment", target=tgt, cfg=self.cfg.sentiment.module
                )
        else:
            logging.info("sentiment excluded!")

        # ----- topic -----
        if self.cfg.topic.active:
            for tgt in self.cfg.topic.target:
                key = f"topic_{tgt}"
                self.runners[key] = build_task_callable(
                    task="topic", target=tgt, cfg=self.cfg.topic.module
                )
        else:
            logging.info("topic excluded!")

        # ----- classifier (sentiment 모델 재사용) -----
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

        # ----- suicide -----
        if hasattr(self.cfg, "suicide") and self.cfg.suicide.active:
            for tgt in self.cfg.suicide.target:
                key = f"suicide_{tgt}"
                self.runners[key] = build_task_callable(
                    task="suicide", target=tgt, cfg=self.cfg.suicide.module
                )
        else:
            logging.info("suicide excluded!")

    # --------------------------------------------------------
    # 🧩 실행
    # --------------------------------------------------------
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


# ============================================================
# 🧪 Test Entry
# ============================================================
if __name__ == "__main__":
    data_dir = "/home/dongwoo38/data/preprocessed_data/2020/01/01-10/news_comments.json"
    data = prepare_data(data_dir)

    runner = AnalysisPipeLine(
        analysis_config=AnalysisConfig.SENT_CMT_TOPIC_TTL(),
        realtime=False,
        monitoring=True,
        save=True,
        save_dir="./outputs",
    )

    result = runner.run(data)
    pprint.pprint(result)
