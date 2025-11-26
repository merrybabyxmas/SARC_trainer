from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Tuple, Optional, Callable, Literal


from .model_config import (
    BaseConfig,
    DefaultModuleConfig,
)

Target = Literal["title", "comments"]

class SummaryUnit(BaseModel):
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


class SummaryConfig(BaseModel):
    classifier : SummaryUnit
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def TTL_SBERT(cls) -> "SummaryConfig":
        mod = BaseConfig(model_name="sbert")
        return cls(
            classifier = SummaryUnit(active=True, target=["title"], module=mod)
        )

    @classmethod
    def CMT_SBERT(cls) -> "SummaryConfig":
        mod = BaseConfig(model_name="sbert")
        return cls(
            classifier = SummaryUnit(active=True, target=["comments"], module=mod)
        )