from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Tuple, Callable, Literal, Optional


class PreProcessConfig(BaseModel):
    included : List[str] = ["title"] #  들어가는 column들 
    
    @field_validator("included")
    @classmethod
    def _validate_included_features(cls, v : List[str]):
        if len(v) < 0:
            raise ValueError("you should include at least 1 element in the PreProcessConfig e.g. 'title', 'comments' ")
        if "title" not in v:
            raise ValueError("you should include title in the LLMconfig")
        return v
    
    @classmethod
    def DefaultConfig(cls) -> "PreProcessConfig":
        return cls(
            included = ["title", "raw_title", "news_date", "comments"]
        )
        
    #TO-DO : 다른 feature들 추가하고 싶다면 아래에 추가
    
    @classmethod
    def Likes_Config(cls) -> "PreProcessConfig":
        return cls(
            included = ["title", "raw_title", "news_date", "comments", "likes"] # 만약 raw data에 likes 가 있다고 가정한다면 
        )
    
