# llm_config.py
from __future__ import annotations
import torch
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Tuple, Optional, Callable, Literal
from .model_config import (
    DefaultModuleConfig
)


LLM_name = Literal["gpt4o", "gpt-opensrc", "gemini", "llama"] # 위 옵션 중 하나 선택 



model_config = DefaultModuleConfig()
SENTIMENT_CLASSES = model_config.sentiment_model.macro_labels
TOPIC_CLASSES = model_config.topic_model.macro_labels



class LLMConfig(BaseModel):
    llm_name    : LLM_name = "gpt-opensrc"
    max_token   : int = 10
    role        : str = ""
    SENTIMENT_CLASSES : List = SENTIMENT_CLASSES
    TOPIC_CLASSES :     List = TOPIC_CLASSES
    
    @field_validator("max_token")
    @classmethod
    def _validate_max_token(cls, v : int):
        if v <0:
            raise ValueError("max token shoule be greater than ")
        
        return v
    
    @classmethod
    def CLASSIFY_SENTIMENT(cls) -> "LLMConfig":
        return cls(
            role = (
            "당신은 한국어 텍스트의 감정을 분석하는 어시스턴트입니다.\n"
            "아래 예시처럼, 입력된 문장이 짧거나 맥락이 부족해도 무조건 6가지 클래스 중 하나로만 답해주세요.\n\n"
            "클래스 목록: " + ", ".join(SENTIMENT_CLASSES) + "\n\n"
            "예시:\n"
            "Input: \"맛집 발견!\"  → Output: 기쁨\n"
            "Input: \"길이 막혀서 늦었어\"  → Output: 불안\n"
            "Input: \"price down\"  → Output: 불안\n"
            "Input: \"title4\"  → Output: 상처   # 실제로는 어떤 울림이 있든 상처로 고정\n\n"
            "이제 분류할 문장: "
        )
            )
        
        
    @classmethod
    def CLASSIFY_TOPIC(cls) -> "LLMConfig":
        return cls(
            role = (
                "당신은 한국어 텍스트의 주제를 분석하는 어시스턴트입니다.\n"
                "아래 예시처럼, 입력된 문장이 짧거나 맥락이 부족해도 무조건 5가지 클래스 중 하나로만 답해주세요.\n\n"
                "클래스 목록: " + ", ".join(TOPIC_CLASSES) + "\n\n"
                "예시:\n"
                "Input: \"코스피 오늘 상승\"  → Output: 경제\n"
                "Input: \"대통령 연설 분석\"  → Output: 정치\n"
                "Input: \"신규 백신 접종률 증가\"  → Output: 생활문화\n"
                "Input: \"title3\"  → Output: 사회   # 실제로는 맥락이 부족해도 ‘사회’로 고정\n\n"
                "이제 분류할 문장: "
            )
            )
        
        
    
    
if __name__ == "__main__":
    config = LLMConfig()
    print(config.CLASSIFY_SENTIMENT().model_dump_json())