from __future__ import annotations
import torch
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, ConfigDict


# ------------------------------------------------------------
# 🧱 BaseConfig — 공통 모델 세팅 (sentiment/topic/classifier/sarcasm)
# ------------------------------------------------------------
class BaseConfig(BaseModel):
    model_name: str
    finetuned: bool = False
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # inference options
    batch_size: int = 2
    truncation: bool = True
    max_length: int = 256
    return_all_scores: bool = False

    # checkpoint path
    ckpt_path: Optional[str] = None

    # optional LoRA / Delta parameters
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    delta_scale: float = 2.0
    use_cls: bool = True

    model_config = ConfigDict(extra="allow")

    @field_validator("batch_size")
    @classmethod
    def _check_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("batch_size must be > 0")
        return v

    @field_validator("max_length")
    @classmethod
    def _check_max_length(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_length must be > 0")
        return v




# ------------------------------------------------------------
# 🧱 BaseModuleConfig — 5 modules (sentiment/topic/classifier/sarcasm/suicide)
# ------------------------------------------------------------
class BaseModuleConfig(BaseModel):
    sentiment_model: BaseConfig
    topic_model: BaseConfig
    classifier_model: BaseConfig
    sarcasm_model: BaseConfig

    # NEW
    suicide_model: BaseConfig

    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------
# 🧩 DefaultModuleConfig — public models preset
# ------------------------------------------------------------
class DefaultModuleConfig(BaseModuleConfig):
    # sentiment model preset
    sentiment_model: BaseConfig = BaseConfig(
        model_name="hun3359/klue-bert-base-sentiment",
        batch_size=32,
        max_length=256,
        return_all_scores=False,
        macro_labels=["분노", "슬픔", "불안", "상처", "당황", "기쁨"],
    )

    # topic model preset
    topic_model: BaseConfig = BaseConfig(
        model_name="freud-sensei/headline_classification",
        batch_size=32,
        max_length=256,
        return_all_scores=False,
        macro_labels=["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"],
    )

    # classifier
    classifier_model: BaseConfig = BaseConfig(
        model_name="MC_SENT_LLFT_0812",
        batch_size=32,
        max_length=256,
        return_all_scores=False,
    )

    # sarcasm model
    sarcasm_model: BaseConfig = BaseConfig(
        model_name="klue/roberta-large",
        finetuned=True,
        batch_size=4,
        truncation=True,
        max_length=128,
        return_all_scores=False,
        ckpt_path="/home/dongwoo38/mindcastlib/assets/sarc_KR.pt",
        r=8,
        alpha=16,
        dropout=0.1,
        delta_scale=2.0,
        use_cls=True,
    )

    # suicide model (global settings)
    suicide_model: BaseConfig = BaseConfig(
        model_name="BM-K/KoSimCSE-roberta-multitask",
        current_date="2022-06",  # YYYY-MM
        sim_weights={
            "token_subtag": 0.5,
            "sent_subtag": 0.2,
            "token_centroid": 0.2,
            "sent_centroid": 0.1
        },
        template="이 문장은 '{subtag}' (키워드: {keyword})에 대한 한국 뉴스 기사 제목이다.",
    )

# ------------------------------------------------------------
# 🔧 Manual test
# ------------------------------------------------------------
if __name__ == "__main__":
    cfg = DefaultModuleConfig()
    print(cfg.model_dump_json(indent=2, exclude_none=True))
