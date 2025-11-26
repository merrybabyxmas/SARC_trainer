import os
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import json


# ------------------------------------------------------------
# 📌 월별 JSON 설정 로더
# ------------------------------------------------------------
def load_suicide_monthly_config(
    date_yyyy_mm: str,
    config_root: str = "/home/yein40/mindcastlib/configs/suicide"
):
    file_key = date_yyyy_mm.replace("-", "_")
    path = os.path.join(config_root, f"{file_key}.json")

    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Suicide monthly config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# ⚙️ BOW Precompute (한국어 subtag → 문장화 임베딩)
# ------------------------------------------------------------
def precompute_bow_embeddings(
    suicide_keywords: Dict[str, List[str]],
    save_path: str,
    tokenizer: AutoTokenizer,
    encoder: AutoModel,
    template: str,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    encoder.eval()

    bow_emb = {}

    with torch.no_grad():
        for kw, subtags in suicide_keywords.items():
            bow_emb[kw] = {}
            for st in subtags:
                text = template.format(subtag=st, keyword=kw)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                outputs = encoder(**inputs)
                token_emb = outputs.last_hidden_state
                sent_emb = token_emb.mean(dim=1).squeeze(0)
                bow_emb[kw][st] = sent_emb.cpu()

    torch.save(bow_emb, save_path)
    print(f"[Precompute] Saved BOW embeddings → {save_path}")


# ------------------------------------------------------------
# 🔍 Suicide Similarity Search Model
# ------------------------------------------------------------
class SimilaritySearchModel(nn.Module):
    """
    - suicide_model BaseConfig 하나만 입력으로 받는다
    - BaseConfig에 포함된(allow extra) 필드에서 설정을 가져온다
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        # ------------------------------------------------
        # 1) 글로벌 suicide 모델 설정(BaseConfig)
        # ------------------------------------------------
        self.model_name = cfg["model_name"]
        self.current_date = cfg.get("current_date")
        if self.current_date is None:
            raise ValueError("[ERROR] BaseConfig must include current_date")

        self.sim_weights = cfg.get("sim_weights", {
            "token_subtag": 0.5,
            "sent_subtag": 0.2,
            "token_centroid": 0.2,
            "sent_centroid": 0.1
        })

        # 정규화
        s = sum(self.sim_weights.values())
        self.sim_weights = {k: v / s for k, v in self.sim_weights.items()}

        self.template = cfg.get(
            "template",
            "이 문장은 '{subtag}' (키워드: {keyword})에 대한 한국 뉴스 기사 제목이다."
        )
        self.bow_root = cfg.get(
            "bow_root",
            "./assets/.precomputed/BagOfWords"
        )

        # ------------------------------------------------
        # 2) 월별 JSON 설정 로드 (keywords, threshold)
        # ------------------------------------------------
        monthly_cfg = load_suicide_monthly_config(self.current_date)
        self.keyword_config = monthly_cfg["keywords"]
        self.default_threshold = monthly_cfg.get("threshold", 0.45)
        self.subtag_thresholds = monthly_cfg.get("subtag_thresholds", {})

        # ------------------------------------------------
        # 3) 토크나이저 / 임베딩 모델
        # ------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.encoder = AutoModel.from_pretrained(self.model_name)
        self.encoder.eval()

        # BOW path
        file_key = self.current_date.replace("-", "_")
        self.bow_path = os.path.join(self.bow_root, f"BOW_{file_key}.pt")

        # ------------------------------------------------
        # 4) BOW 로드 or 생성
        # ------------------------------------------------
        self.bow_emb = self._load_or_precompute()

    # --------------------------------------------------------
    def _load_or_precompute(self):
        if os.path.exists(self.bow_path):
            print(f"[Load] BOW Loaded from {self.bow_path}")
            return torch.load(self.bow_path)

        print(f"[Warn] {self.bow_path} not found → Precomputing...")
        precompute_bow_embeddings(
            suicide_keywords=self.keyword_config,
            save_path=self.bow_path,
            tokenizer=self.tokenizer,
            encoder=self.encoder,
            template=self.template,
        )
        return torch.load(self.bow_path)

    # --------------------------------------------------------
    def _encode_titles(self, titles: List[str]):
        inputs = self.tokenizer(titles, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = self.encoder(**inputs)
            token_emb = outputs.last_hidden_state          # (B, L, H)
            sent_emb = token_emb.mean(dim=1)               # (B, H)

        return token_emb, sent_emb, inputs["attention_mask"]

    # --------------------------------------------------------
    def _prepare_bow_tensors(self):
        subtag_list = []
        keyword_of_subtag = []
        subtag_embs = []

        # keyword centroid
        kw_to_embs = {kw: [emb for emb in subtags.values()] for kw, subtags in self.bow_emb.items()}
        kw_centroid = {
            kw: torch.stack(embs, dim=0).mean(dim=0)
            for kw, embs in kw_to_embs.items()
        }

        centroid_list = []

        for kw, subtags in self.bow_emb.items():
            centroid = kw_centroid[kw]
            for st, emb in subtags.items():
                subtag_list.append(st)
                keyword_of_subtag.append(kw)
                subtag_embs.append(emb.unsqueeze(0))
                centroid_list.append(centroid.unsqueeze(0))

        bow_tensor = torch.cat(subtag_embs, dim=0)
        centroid_tensor = torch.cat(centroid_list, dim=0)

        return bow_tensor, centroid_tensor, subtag_list, keyword_of_subtag


    # --------------------------------------------------------
    # 🔥 Final integrated forward()
    # --------------------------------------------------------
    def forward(self, titles: List[str]) -> List[Dict[str, Any]]:
        device = next(self.encoder.parameters()).device

        # 1) Encode titles
        token_emb, sent_emb, attn_mask = self._encode_titles(titles)
        token_emb = token_emb.to(device)
        sent_emb = sent_emb.to(device)
        attn_mask = attn_mask.to(device)

        B, L, H = token_emb.shape

        # 2) Prepare data
        bow_tensor, centroid_tensor, subtag_list, keyword_of_subtag = self._prepare_bow_tensors()
        bow_tensor = bow_tensor.to(device)
        centroid_tensor = centroid_tensor.to(device)
        M = bow_tensor.size(0)

        # 3) Normalize
        token_norm = F.normalize(token_emb, dim=-1)
        sent_norm = F.normalize(sent_emb.unsqueeze(1), dim=-1).squeeze(1)
        bow_norm = F.normalize(bow_tensor, dim=-1)
        centroid_norm = F.normalize(centroid_tensor, dim=-1)

        # 4) token similarities
        sim_token_subtag = torch.einsum("blh,mh->blm", token_norm, bow_norm)
        sim_token_centroid = torch.einsum("blh,mh->blm", token_norm, centroid_norm)

        mask = attn_mask.unsqueeze(-1).bool()
        sim_token_subtag = sim_token_subtag.masked_fill(~mask, -1e4)
        sim_token_centroid = sim_token_centroid.masked_fill(~mask, -1e4)

        max_token_subtag = sim_token_subtag.max(dim=1).values   # (B, M)
        max_token_centroid = sim_token_centroid.max(dim=1).values

        # 5) sentence sim
        sim_sent_subtag = torch.matmul(sent_norm, bow_norm.t())
        sim_sent_centroid = torch.matmul(sent_norm, centroid_norm.t())

        # 6) final weighted similarity
        w = self.sim_weights
        final_sim = (
            w["token_subtag"] * max_token_subtag +
            w["sent_subtag"] * sim_sent_subtag +
            w["token_centroid"] * max_token_centroid +
            w["sent_centroid"] * sim_sent_centroid
        )

        final_sim_cpu = final_sim.cpu()

        unique_keywords = list(set(keyword_of_subtag))

        CENTROID_THRESHOLD = 0.20
        LOW_THRESHOLD = 0.25

        results = []

        # --------------------------------------------------------
        # Per title inference
        # --------------------------------------------------------
        for b in range(B):

            keyword_avg_sim = {}
            for kw in unique_keywords:
                idxs = [i for i, k in enumerate(keyword_of_subtag) if k == kw]
                keyword_avg_sim[kw] = final_sim_cpu[b, idxs].mean().item()

            # completely unrelated
            if max(keyword_avg_sim.values()) < LOW_THRESHOLD:
                keyword_mask = {kw: False for kw in unique_keywords}
                subtag_mask = {st: False for st in subtag_list}
                results.append({
                    "title": titles[b],
                    "suicide_related": False,
                    "keyword_mask": keyword_mask,
                    "subtag_mask": subtag_mask,
                    "winner_keyword": None,
                })
                continue

            # centroid filter
            filtered = {
                kw: avg for kw, avg in keyword_avg_sim.items()
                if avg >= CENTROID_THRESHOLD
            }
            if not filtered:
                filtered = keyword_avg_sim

            winner_kw = max(filtered, key=filtered.get)

            # winner subtag mask
            keyword_mask = {}
            subtag_mask = {}

            for kw in unique_keywords:
                keyword_mask[kw] = False

            for j, st in enumerate(subtag_list):
                kw = keyword_of_subtag[j]

                if kw != winner_kw:
                    subtag_mask[st] = False
                    continue

                thr = self.subtag_thresholds.get(st, self.default_threshold)
                subtag_mask[st] = final_sim_cpu[b, j].item() > thr

            winner_has_active = any(
                subtag_mask[st]
                for st, kw in zip(subtag_list, keyword_of_subtag)
                if kw == winner_kw
            )
            if winner_has_active:
                keyword_mask[winner_kw] = True

            suicide_related = any(keyword_mask.values())

            results.append({
                "title": titles[b],
                "suicide_related": suicide_related,
                "keyword_mask": keyword_mask,
                "subtag_mask": subtag_mask,
                "winner_keyword": winner_kw if suicide_related else None,
            })

        return results


# ------------------------------------------------------------
# 🔬 Example 실행
# ------------------------------------------------------------
if __name__ == "__main__":
    from mindcastlib.configs import DefaultModuleConfig

    suicide_cfg = DefaultModuleConfig().suicide_model.model_dump()

    model = SimilaritySearchModel(suicide_cfg)

    sample_texts = [
        "청년 실업률이 급증하고 있다",
        "주식 시장이 상승 중이다",
        "가계부채와 빚 문제가 심각하다"
    ]

    results = model(sample_texts)

    for r in results:
        print("\n====================")
        print(r)
