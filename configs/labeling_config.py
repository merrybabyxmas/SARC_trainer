import torch
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Tuple, Optional, Callable, Literal


from .llm_config import LLMConfig

Target = Literal["title", "comments"]

class LabelingUnit(BaseModel):
    active: bool = True
    target: List[Target] = Field(default_factory=["title"])
    llm   : LLMConfig
    
    model_config = ConfigDict(extra="forbid")

    
    @field_validator("target")
    @classmethod
    def _validate_target(cls, v: List[Target]) -> List[Target]:
        if not v:
            raise ValueError("target must contain at least one of 'title' or 'comments'")
        # 중복 제거
        v = list(dict.fromkeys(v))
        return v
    
    
class LabelingConfig(BaseModel):
    
    sentiment: LabelingUnit
    topic    : LabelingUnit
    
    model_config = ConfigDict(extra="forbid")

    # ------------presets(Default)------------
    @classmethod
    def SENT_CMT_TOPIC_TTL(cls) -> "LabelingConfig":
        lc = LLMConfig() # llm confing
        return cls(
            sentiment = LabelingUnit(active=True, target=["comments"], llm= lc.CLASSIFY_SENTIMENT()),
            topic = LabelingUnit(active=True, target = ["title"], llm = lc.CLASSIFY_TOPIC())
        )
        
        
    @classmethod
    def SENT_ALL_TOPIC_TTL(cls) -> "LabelingConfig":
        lc = LLMConfig() # llm confing
        return cls(
            sentiment = LabelingUnit(active=True, target=["title", "comments"], llm= lc.CLASSIFY_SENTIMENT()),
            topic = LabelingUnit(active=True, target = ["title"], llm = lc.CLASSIFY_TOPIC())
        )
        
    @classmethod
    def SENT_ONLY_CMT(cls) -> "LabelingConfig":
        lc = LLMConfig() # llm confing
        return cls(
            sentiment = LabelingUnit(active=True, target=["comments"], llm= lc.CLASSIFY_SENTIMENT()),
            topic = LabelingUnit(active=False, target = ["title"], llm = lc.CLASSIFY_TOPIC())
        )
    
    @classmethod
    def TOPIC_ONLY_TTL(cls) -> "LabelingConfig":
        lc = LLMConfig() # llm confing
        return cls(
            sentiment = LabelingUnit(active=False, target=["comments"], llm= lc.CLASSIFY_SENTIMENT()),
            topic = LabelingUnit(active=True, target = ["title"], llm = lc.CLASSIFY_TOPIC())
        )
    