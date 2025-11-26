import torch 
from mindcastlib.src import LLMPipeLine
from typing import List, Dict, Tuple, Literal
from sklearn.metrics import precision_recall_fscore_support, classification_report
from mindcastlib.configs import EvaluationConfig, LLMConfig, BaseModuleConfig, DefaultModuleConfig

llm_cfg = LLMConfig()
model_cfg = BaseModuleConfig

class _EvaluationPipeLine:
    def __init__(self, cfg: EvaluationConfig | None = None):
        # 누락된 __init__ 복구 + 기본값 처리
        self.cfg = cfg or EvaluationConfig.DefaultConfig()

    def __call__(self, preds: List[str], trues: List[str]) -> Dict:
        # print(f"preds : {preds}")
        # print(f"trues : {trues}")
        data_len = len(preds)
        
        # average는 EvaluationConfig 소속, zero_division은 pred 설정 사용(요구시 true로 변경 가능)
        precision, recall, f1, _ = precision_recall_fscore_support(
            trues,
            preds,
            average=self.cfg.average,
            labels=self.cfg.labels,
            zero_division=self.cfg.pred.zero_division,
        )
        report = classification_report(
            trues,
            preds,
            labels=self.cfg.labels,
            zero_division=self.cfg.pred.zero_division,
            output_dict= True
        )
        return {
            "llm_name" : llm_cfg.llm_name,
            "target" : self.cfg.target,
            "task" : self.cfg.task,
            "data length" : data_len, 
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "detail_report": report,
        }


        
        
        



        