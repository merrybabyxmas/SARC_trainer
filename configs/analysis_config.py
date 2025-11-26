from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict

from .model_config import (
    BaseConfig,
    DefaultModuleConfig,
)

Target = Literal["title", "comments"]


class AnalysisUnit(BaseModel):
    active: bool = True
    target: List[Target] = Field(default_factory=lambda: ["title"])
    module: BaseConfig

    model_config = ConfigDict(extra="forbid")

    @field_validator("target")
    @classmethod
    def _validate_target(cls, v: List[Target]) -> List[Target]:
        if not v:
            raise ValueError("target must contain at least one of 'title' or 'comments'")
        # 중복 제거
        v = list(dict.fromkeys(v))
        return v


class AnalysisConfig(BaseModel):
    sentiment: AnalysisUnit
    topic: AnalysisUnit
    sarcasm: AnalysisUnit
    classifier: AnalysisUnit
    suicide: AnalysisUnit

    model_config = ConfigDict(extra="forbid")

    # -------- presets(Default)--------
    @classmethod
    def SENT_CMT_TOPIC_TTL(cls) -> "AnalysisConfig":
        mods = DefaultModuleConfig()
        return cls(
            sentiment=AnalysisUnit(active=True, target=["comments"], module=mods.sentiment_model),
            topic=AnalysisUnit(active=True, target=["title"], module=mods.topic_model),
            sarcasm=AnalysisUnit(active=False, target=["comments"], module=mods.sarcasm_model),  # ✅ 추가
            classifier=AnalysisUnit(active=False, target=["comments"], module=mods.classifier_model),
            suicide=AnalysisUnit(active=True, target=["title"], module=mods.suicide_model)
        )

    @classmethod
    def SENT_TTL_TOPIC_CMT(cls) -> "AnalysisConfig":
        mods = DefaultModuleConfig()
        return cls(
            sentiment=AnalysisUnit(active=True, target=["title"], module=mods.sentiment_model),
            topic=AnalysisUnit(active=True, target=["comments"], module=mods.topic_model),
            sarcasm=AnalysisUnit(active=False, target=["comments"], module=mods.sarcasm_model),  # ✅ 추가
            classifier=AnalysisUnit(active=True, target=["comments"], module=mods.classifier_model),
        )

    @classmethod
    def SENT_ALL_TOPIC_ALL(cls) -> "AnalysisConfig":
        mods = DefaultModuleConfig()
        both = ["title", "comments"]
        return cls(
            sentiment=AnalysisUnit(active=True, target=both, module=mods.sentiment_model),
            topic=AnalysisUnit(active=True, target=both, module=mods.topic_model),
            sarcasm=AnalysisUnit(active=False, target=both, module=mods.sarcasm_model),  # ✅ 추가
            classifier=AnalysisUnit(active=False, target=both, module=mods.classifier_model),
        )


if __name__ == "__main__":
    cfg = AnalysisConfig.SENT_CMT_TOPIC_TTL()
    print(cfg.model_dump_json(indent=2, exclude_none=True))
    for name in cfg.model_fields:  # sentiment, topic, sarcasm, classifier
        unit = getattr(cfg, name)
        print(name, unit.active, unit.target)
        if unit.active:
            print("yes")
