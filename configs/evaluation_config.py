from __future__ import annotations
import torch 
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Dict, Tuple, Union
from .model_config import DefaultModuleConfig


# 빈 리스트면 스코어 계산이 불가하므로 기본값 하나라도 둠 (원하면 바꿔서 사용)
sent_config = DefaultModuleConfig()
SENT_LABELS = sent_config.sentiment_model.macro_labels

Task = Literal["sentiment", "topic", "summary"]
Target = Literal["title", "comments"]


class EvaluationUnit(BaseModel):
    target: str
    zero_division: int = 0   # 오탈자(zero_divison) 수정


class EvaluationConfig(BaseModel):
    target : Target
    task : Task
    pred: EvaluationUnit
    true: EvaluationUnit
    labels: List[str]
    average: str = "macro"

    @classmethod
    def SENT_CMT_ONLY(cls):
        return cls(
            target = "comments",
            task = "sentiment",
            pred=EvaluationUnit(target="SentimentClassificationPipeLine_comments"),
            true=EvaluationUnit(target="LLMPipeLine_comments"),
            labels=SENT_LABELS,
        )

    @classmethod
    def DefaultConfig(cls):
        return cls.SENT_CMT_ONLY()
