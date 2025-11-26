### TODO: [대원] 코드 추가 
import os
from typing import List, Tuple, Dict, Optional, Literal
from mindcastlib.configs import SummaryConfig

import torch

class SummaryPipeLine:
    def __init__(self, cfg: SummaryConfig):
        # config 파일 확인해서 이에 맞춰서 파이프라인 설계할 것 
        pass
    def forward(self, data:List[str]) -> List[str]:
        # 위의 형식 맞춰서 결과 도출할 것 
        pass
    
    
if __name__ == "__main__":
    cfg_title_sbert = SummaryConfig.TTL_SBERT()
    cfg_comments_sbert = SummaryConfig.CMT_SBERT()
    # config 예시 출력 
    print(cfg_title_sbert.model_dump_json(indent=2, exclude=None))
    print(cfg_comments_sbert.model_dump_json(indent=2, exclude=None))
    
    
    # 데이터 형식은 모두 List[str] 형식으로 통일할 것 
    title_data = ["물가가 올랐다", "관세 폭탄으로 인해 물가 변동"] # 예시 데이터 형식임. 실제로는 이렇지 않음
    comments_data = ["이래서 1찍들이 안돼", "전광훈 만세"]
    
    title_pipe = SummaryPipeLine(cfg_title_sbert)
    comments_pipe = SummaryPipeLine(cfg_comments_sbert)
    
    
    res_title = title_pipe(title_data)
    res_comments = comments_pipe(comments_data)
    
    