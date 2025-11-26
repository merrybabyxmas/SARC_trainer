from mindcastlib.src import (
    _EvaluationPipeLine,
    prepare_data,
    apply_func_to_something_from_commentlike_double_data as apply_double_cmt_likedata,
    apply_func_to_something_from_titlelike_double_data as apply_double_ttl_likedata,
)
from mindcastlib.configs import EvaluationConfig


from typing import List, Dict, Literal
from pydantic import BaseModel
from sklearn.metrics import precision_recall_fscore_support, classification_report
from dataclasses import dataclass
import logging
import time
import json
import os


Task = Literal["sentiment", "topic", "summary"]
Target = Literal["title", "comments"]

@dataclass
class MCEvalSpec:
    task: Literal["sentiment", "topic", "summary"]
    target: Literal["title", "comments"]
    cfg: EvaluationConfig


class ModuleCallable:
    def __init__(self, spec: MCEvalSpec):
        self.task = spec.task
        self.target = spec.target
        self.cfg = spec.cfg
        self.pipe = _EvaluationPipeLine(self.cfg)

    def __call__(self, true: Dict, pred: Dict):
        if len(true) <= 0 or len(pred) <= 0:   # 비교 연산 수정
            raise ValueError("true or pred should be greater than 0")

        if self.target == "title":
            return apply_double_ttl_likedata(func=self.pipe, data1=true, data2=pred,
                                             target1= self.cfg.true.target,
                                             target2= self.cfg.pred.target,
                                             labels_only=True)
        elif self.target == "comments":
            return apply_double_cmt_likedata(func=self.pipe, data1=true, data2=pred,
                                             target1= self.cfg.true.target,
                                             target2= self.cfg.pred.target,
                                             labels_only=True)

        # 기본 반환 (실행 도달 X)
        return {"warning": "unknown target", "true": true, "pred": pred}


def build_task_callable(task: Task, target: Target, cfg: EvaluationConfig) -> ModuleCallable:
    return ModuleCallable(MCEvalSpec(task=task, target=target, cfg=cfg))


class EvaluationPipeLine:
    def __init__(
        self,
        pred_data: Dict | None = None,
        true_data: Dict | None = None,
        evaluation_config: EvaluationConfig | None = None,
        realtime: bool | None = None,
        monitoring: bool | None = None,
        save: bool | None = None,
        save_dir: str | None = None,
    ):
        # 필드 초기화 누락 보완
        self.pred_data = pred_data
        self.true_data = true_data
        self.cfg = evaluation_config or EvaluationConfig.DefaultConfig()
        self.realtime = bool(realtime) if realtime is not None else False
        self.monitoring = bool(monitoring) if monitoring is not None else False
        self.save = bool(save) if save is not None else False
        self.save_dir = save_dir or "./outputs/evaluation"
        os.makedirs(self.save_dir, exist_ok=True)

        self._load_models()

    ### 수정필요
    def _load_models(self):
        # 원래 구조(여러 runner) 유지하되, 현재 cfg로부터 title/comments 감지해서 러너 구성
        self.runners: Dict[str, ModuleCallable] = {}

        # target 문자열에 'title' / 'comments' 포함 여부로 라우팅 (사용자 설계 존중)
        want_title = ("title" in self.cfg.pred.target) or ("title" in self.cfg.true.target)
        want_comments = ("comments" in self.cfg.pred.target) or ("comments" in self.cfg.true.target)

        if want_title:
            self.runners["eval_title"] = build_task_callable(task="sentiment", target="title", cfg=self.cfg)
        if want_comments:
            self.runners["eval_comments"] = build_task_callable(task="sentiment", target="comments", cfg=self.cfg)

        # 아무것도 감지 못했을 때는 comments 기준으로 하나라도 만들어 줌(안전장치)
        if not self.runners:
            self.runners["eval_comments"] = build_task_callable(task="sentiment", target="comments", cfg=self.cfg)
            if self.monitoring:
                logging.info("[_load_models] No explicit target found; defaulting to comments")

    def run(self, true: Dict, pred: Dict):
        t0 = time.time()
        results: Dict[str, Dict] = {}

        for key, runner in self.runners.items():
            if self.monitoring:
                logging.info(f"[RUN] executing runner: {key}")
            results[key] = runner(true, pred)

        if self.save:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(self.save_dir, f"infer_{ts}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            if self.monitoring:
                logging.info(f"[Inference] saved to {out_path}")

        if self.monitoring:
            logging.info(f"[RUN] done in {time.time() - t0:.2f}s")

        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    tc = ["2023-05-01", "2023-05-02"]
    pred_dir = "/home/dongwoo38/outputs/analysis/infer_20250820_145734.json"
    true_dir = "/home/dongwoo38/outputs/labeled/labeling_20250820_145921.json"

    data_pred = prepare_data(data_dir=pred_dir)
    
    data_true = prepare_data(data_dir=true_dir)

    cfg = EvaluationConfig.DefaultConfig()

    pipeline = EvaluationPipeLine(
        evaluation_config=cfg,
        realtime=False,   # evaluation의 경우, realtime은 항상 False
        monitoring=True,
        save=True,
        save_dir="./outputs/evaluation",
    )

    pipeline.run(data_true, data_pred)