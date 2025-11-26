# emotion_utils.py
from typing import List, Sequence, Union
import numpy as np
import torch




###########################################
###########################################
### 감정 분류 관련 utils 
###########################################
###########################################

def should_collapse6(task: str, model) -> bool:
    """모델이 60-way 감정 분류이고 task가 sentiment면 6대분류 집계를 활성화."""
    return task == "sentiment" and getattr(model.config, "num_labels", None) == 60

def macro_slices_60x6() -> List[slice]:
    """60라벨을 6대분류(각 10개)로 묶는 인덱스 슬라이스."""
    return [
        slice(0, 10),   # 분노
        slice(10, 20),  # 슬픔
        slice(20, 30),  # 불안
        slice(30, 40),  # 상처
        slice(40, 50),  # 당황
        slice(50, 60),  # 기쁨
    ]

def predict_6sentiments(
    texts: Union[str, List[str]],
    pipe,
    truncation: bool,
    max_length: int,
    return_all_scores: bool,
    macro_labels: Sequence[str] = ("분노", "슬픔", "불안", "상처", "당황", "기쁨"),
    macro_slices: Sequence[slice] | None = None,
    **kwargs
):
    """
    HF pipeline의 tokenizer/model을 직접 써서 60-way 확률을 얻은 뒤 6대분류로 합산.
    반환 형식은 pipeline과 호환(단일/배치 모두 지원).
    """
    single = isinstance(texts, str)
    batch_texts = [texts] if single else texts

    tok = pipe.tokenizer
    mdl = pipe.model
    dev = mdl.device

    enc = tok(
        batch_texts,
        truncation=truncation,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    ).to(dev)

    with torch.no_grad():
        logits = mdl(**enc).logits  # [B,60]
        probs60 = torch.softmax(logits, dim=-1).cpu().numpy()  # [B,60]

    if macro_slices is None:
        macro_slices = macro_slices_60x6()

    # 6개 그룹 합산 (행 합 1 유지)
    macro = np.stack([probs60[:, s].sum(axis=1) for s in macro_slices], axis=1)  # [B,6]

    if return_all_scores:
        out = []
        for row in macro:
            items = [{"label": lb, "score": float(sc)} for lb, sc in zip(macro_labels, row)]
            items.sort(key=lambda x: x["score"], reverse=True)
            out.append(items)
    else:
        out = []
        for row in macro:
            k = int(row.argmax())
            out.append([{"label": macro_labels[k], "score": float(row[k])}])

    return out[0] if single else out





###########################################
###########################################
### sarcasm 관련 utils 
###########################################
###########################################