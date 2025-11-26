
# mindcastlib/src/train_utils.py
# -*- coding: utf-8 -*-

"""
List[str] -> List[str] 형태로 LLM 기반 분류를 수행하는 통합 유틸.

- 지원 백엔드(llm_name): "gpt4o" | "gpt-opensrc" | "gemini" | "llama"
- Config는 mindcastlib.configs.LLMConfig를 그대로 사용
    -> 우선 최대한 동우 선배님이 작성한 기본 LLMconfig Role을 활용했습니다. (config 자체는 수정 안 하는 방향으로)
- 각 백엔드는 내부적으로 적합한 프롬프트/템플릿으로 변환해 라벨만 반환하도록 설계
    -> 이 유틸 목적이 데이터 라벨링인 것 같아서, 우선 설계 자체를 라벨값만 반환하도록 했습니다.

필수/선택 패키지 요약:
- 공통: pydantic(LLMConfig), typing
- OpenAI: openai  / 환경변수: OPENAI_API_KEY
- Gemini: google-generativeai / 환경변수: GEMINI_API_KEY (또는 GOOGLE_API_KEY)
- Hugging Face: transformers (zero-shot-classification 파이프라인)
- * LLaMA: transformers(text-generation), 공개 모델 기본값(TinyLlama)
-> 일단 개인 API를 호출해서 모든 모델 실행 가능 여부 확인 완료했습니다.
    근데 llama는 계속,, 라벨링을 제대로 하지 못해서(bias 발생,,), 조금 더 만지고 있습니다.

주의:
- 어떤 백엔드도 실패해도 프로세스가 죽지 않도록, 요소별 에러 문자열을 반환합니다.
- 생성형(OpenAI/Gemini/LLaMA)은 프롬프트+후처리로 "라벨만" 출력되게 제약.
- HF zero-shot은 후보 라벨 중 1개를 안정적으로 반환(후처리 필요 없음).
    -> Hugging Face는 생성형 모델, zero-shot 모두 해봤는데 생성형 모델은 출력값이 이상 + 빈칸 반환의 경우가 너무 많아서
       zero-shot으로 교체해봤습니다. (이후 수정 필요하면 말씀 주세요!)

라벨링 결과,
정확도 의 경우 gemini > hugging face > gpt4o > llama 순인 것 같습니다.

그리고 llm_config에서 Topic이 5개밖에 없는데, KLUE-TC TOPIC 분류 체계는 7개더라구요.
살짝 수정해서 라벨링 했더니, 토픽 분류 라벨링도
괜찮게 작동하는 것 같습니다.

"""

from __future__ import annotations
from typing import List, Dict, Callable
import os
import re
import warnings
from mindcastlib.configs import LLMConfig

#from mindcastlib.configs.llm_config import LLMConfig


# ---------------------------------------------------------------------
# [Secrets Utils] .env 로드/저장 (python-dotenv 없으면 수동 폴백)
# ---------------------------------------------------------------------
_DOTENV_PATH = os.environ.get("MINDCASTLIB_DOTENV", ".env")

def load_api_keys(env_path: str = _DOTENV_PATH) -> None:
    """
    .env 파일을 읽어 현재 프로세스의 환경변수(os.environ)에 로드한다.

    동작:
    - python-dotenv가 설치되어 있으면 load_dotenv 사용.
    - 없으면 수동으로 KEY=VALUE 라인을 파싱해 os.environ에 setdefault.

    주의:
    - 이미 설정된 환경변수는 override=False 정책으로 덮지 않는다(명시적 설정 우선).
    - 존재하지 않거나 읽기 실패하더라도 조용히 패스(런타임에서 prompt_and_save_if_missing로 보완 가능).

    현재 gpt4o는 토큰을 다 써서 작동이 안 되고,  hugging face, gemini의 API key는 잘 작동합니다!
    """

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except Exception:
        # 수동 폴백 로드
        if not os.path.exists(env_path):
            return
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            pass

def save_api_key(var: str, value: str, env_path: str = _DOTENV_PATH) -> bool:
    """
    단일 환경변수를 메모리(os.environ)와 .env 파일에 저장한다.

    매개변수:
    - var: 환경변수 이름 (e.g., "OPENAI_API_KEY")
    - value: 저장할 값(공백 제거 후 빈 문자열이면 저장 안 함)
    - env_path: .env 파일 경로 (기본값: 프로젝트 루트의 .env)

    구현 디테일:
    - python-dotenv가 있으면 set_key로 안전하게 갱신.
    - 없으면 기존 파일을 읽어 해당 var 라인을 제거한 후 마지막에 새 KEY=VALUE 라인을 추가.

    반환:
    - True: 파일 저장 성공 (또는 set_key로 성공)
    - False: 파일 저장 실패(단, os.environ에는 이미 반영됨)
    """

    value = (value or "").strip()
    if not value:
        return False
    # 먼저 환경변수에 반영
    os.environ[var] = value
    # .env 저장 (python-dotenv 있으면 set_key 사용)
    try:
        from dotenv import set_key
        set_key(env_path, var, value)
        return True
    except Exception:
        # 수동 쓰기
        try:
            lines = []
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    lines = [ln.rstrip("\n") for ln in f.readlines()]
                # 기존 키 라인 제거
                lines = [ln for ln in lines if not ln.startswith(f"{var}=")]
            lines.append(f"{var}={value}")
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            return True
        except Exception:
            return False

def prompt_and_save_if_missing(var: str, label: str) -> None:
    """
    특정 환경변수가 비어 있을 경우, 사용자에게 안전하게 입력을 받아 .env와 os.environ에 저장한다.

    매개변수:
    - var: 환경변수 이름
    - label: 사용자 프롬프트에 보여줄 안내 라벨(키 이름 친절 표기)

    동작:
    - getpass로 마스킹 입력을 받아 유출을 방지.
    - 입력이 비어 있으면 스킵, 값이 있으면 save_api_key로 파일+환경 동시 갱신.

    주의:
    - 비대화형 환경(예: 일부 CI/IDE)에서는 getpass가 실패할 수 있으며, 그 경우 조용히 패스.
    """

    if os.getenv(var):
        return
    try:
        from getpass import getpass
        v = getpass(f"Enter {label} (leave empty to skip): ").strip()
        if v:
            if save_api_key(var, v):
                print(f"[secrets] {var} saved to {_DOTENV_PATH} and exported to env.")
            else:
                print(f"[secrets] Failed to save {var} to {_DOTENV_PATH}, but exported to env.")
        else:
            print(f"[secrets] Skipped {var}.")
    except Exception:
        # getpass 사용 불가 환경(예: 일부 IDE)에서는 그냥 패스
        pass

# torch: GPU 유무 체크 및 transformers 최적화용(없어도 동작)
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

ALLOWED_LLM_NAMES = ("gpt4o", "gpt-opensrc", "gemini") #"llama"


# ---------------------------------------------------------------------
# RoleAdapter: cfg.role을 해석해 백엔드별 프롬프트/템플릿/라벨 유틸 제공
# ---------------------------------------------------------------------
class RoleAdapter:
    """
    LLMConfig.role 내용을 해석해 백엔드 친화적인 지시문/템플릿/라벨 유틸을 제공하는 어댑터.

    설계 배경:
    - 같은 과업(감정/주제 분류)이라도 백엔드마다 선호하는 입출력 방식이 다르다.
    * 생성형(OpenAI/Gemini/LLaMA): 지시문(prompt)을 읽고 텍스트 생성 → "라벨만" 출력하도록 강한 제약 필요
    * HF zero-shot(NLI): hypothesis_template + candidate_labels로 분류 → role 문자열은 라벨셋 결정에만 활용
    : 사실 config 내부에서 주요 role은 크게 바뀌지 않을 것이라고 생각합니다.(보편적 규칙 적용)
      그래서 생성형 모델 별로 role을 더 추가해, 원하는 라벨 값을 더 잘 얻을 수 있도록 시도해봤습니다! 만약 필요 없다면, 기존에 RoleAdapter가 없는 버전도 있어서 바로 수정 가능합니다!
    - RoleAdapter는 이 차이를 감추고, 각 caller가 동일한 방식으로 프롬프트/템플릿을 받아 쓰도록 (이후 많은 데이터들에 대한 라벨링 용이성을 위함)
        하는 역할입니다.
    - 만약, 라벨링 자체가 목적이 아니라면 RoleAdapter가 없는 버전을 사용해 output 값을 그냥 받아올 수 있습니다!
      다양하게 시도해봤는데, 이렇게 제약 조건을 좀 강하게 둬야지 라벨링 결과가 잘 나오더라구요..!

    주요 속성:
    - task            : "감정" 또는 "주제" (role 문자열의 키워드로 판별, 기본은 "감정")
    - labels          : 후보 라벨 리스트 (SENTIMENT_CLASSES / TOPIC_CLASSES에서 가져옴)
    - label_line      : 쉼표로 연결한 라벨 문자열 (생성형 프롬프트용)
    - idx2label       : "1"→"분노" 같은 숫자→라벨 매핑(LLaMA에서 숫자만 출력 유도)
    - numbered        : "1) 분노; 2) 슬픔; ..." (LLaMA 안내문에 사용)

    주요 메서드:
    - generative_prompt(text): OpenAI/Gemini용 "라벨만" 출력 프롬프트 생성
    - llama_numeric_prompt(text): LLaMA용 "숫자만" 출력 프롬프트 생성(첫 라벨 바이어스 제거)
    - zshot_template(): HF zero-shot의 hypothesis_template 반환
    - pick_label(pred): 생성형 결과 문자열에서 라벨만 안전하게 추출(완전일치→부분일치→토큰일치)
    """

    def __init__(self, cfg: LLMConfig, include_cfg_role_prefix: bool = False):
        self.cfg = cfg
        self.include_prefix = include_cfg_role_prefix
        #기본값 자체는 False
        #include_prefix=True → generative_prompt()가 cfg.role을 프리픽스로 붙임
        #include_prefix=False → cfg.role은 붙지 않고, 짧은 지시문(_gen_instr + label_line)만 사용

        role = (cfg.role or "")
        if "감정" in role:
            self.task = "감정"
            self.labels = list(getattr(cfg, "SENTIMENT_CLASSES", ["분노", "슬픔", "불안", "상처", "당황", "기쁨"]))
            self._zshot_template = "이 문장의 감정은 {}이다."
            self._gen_instr = "아래 문장의 감정을 다음 후보 중 하나로만 출력해:"
        elif "주제" in role:
            self.task = "주제"
            self.labels = list(getattr(cfg, "TOPIC_CLASSES", ["경제", "정치", "사회", "연예", "건강","문화","세계","IT/과학","스포츠"]))
            self._zshot_template = "이 문장의 주제는 {}이다."
            self._gen_instr = "아래 문장의 주제를 다음 후보 중 하나로만 출력해:"
        else:
            self.task = "감정"
            self.labels = ["분노", "슬픔", "불안", "상처", "당황", "기쁨"]
            self._zshot_template = "이 문장의 감정은 {}이다."
            self._gen_instr = "아래 문장의 감정을 다음 후보 중 하나로만 출력해:"

        self.label_line = ", ".join(self.labels)
        self.idx2label = {str(i + 1): lb for i, lb in enumerate(self.labels)}
        self.numbered = "; ".join([f"{i}) {lb}" for i, lb in self.idx2label.items()])

    def generative_prompt(self, text: str) -> str:
        """OpenAI/Gemini 등 생성형 모델에 주는 프롬프트(라벨만 출력하도록 강제)."""
        prefix = f"{self.cfg.role}\n\n" if self.include_prefix and self.cfg.role else ""
        return (
            f"{prefix}{self._gen_instr} {self.label_line}\n"
            f"- 반드시 후보 중 하나만, 아무 설명 없이 한 단어로 출력.\n"
            f"- 출력 형식: 라벨만.\n\n"
            f"문장: {text}\n라벨:"
        )

    def llama_numeric_prompt(self, text: str) -> str:
        """LLaMA용 프롬프트: 숫자 한 글자만 출력하게 유도(생성형 바이어스/장황출력 방지)."""
        return (
            f"다음 한국어 문장의 {self.task}를 번호로만 답하라.\n"
            f"라벨 후보: {self.numbered}\n"
            f"- 출력 형식: 오직 숫자 한 글자 (예: 1)\n"
            f"- 추가 문자, 공백, 설명 금지\n\n"
            f"문장: {text}\n정답 번호:"
        )

    def zshot_template(self) -> str:
        """HF zero-shot classification의 hypothesis_template 반환."""
        return self._zshot_template

    def pick_label(self, pred: str) -> str:
        """
        생성형 결과 문자열에서 config 파일에서 제한한 감정, 주제 라벨만 추출한다.
        우선순위: 완전일치 → 부분일치 → 토큰 단위 일치. 실패 시 빈 문자열 반환.
        """
        s = (pred or "").strip()
        for lb in self.labels:
            if s == lb:
                return lb
        for lb in self.labels:
            if lb in s:
                return lb
        for t in s.replace(",", " ").split():
            if t in self.labels:
                return t
        return ""


# ---------------------------------------------------------------------
# Caller 인터페이스
# ---------------------------------------------------------------------
class BaseCaller:
    """
    Caller의 최소 인터페이스.

    역할:
    - 모든 백엔드 caller는 BaseCaller를 상속하고, __call__(List[str]) -> List[str]를 구현한다.
    - self.adapter(RoleAdapter)를 통해 라벨/프롬프트/템플릿 공통 유틸을 재사용한다.
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.adapter = RoleAdapter(cfg) # Boolean으로 RoleAdapter 값 설정


    def __call__(self, data: List[str]) -> List[str]:
        """
        파생 클래스에서 구현해야 하는 핵심 메서드.
        입력: List[str]
        출력: List[str] (입력과 동일한 길이로, 각 요소는 최종 라벨 문자열)
        """
        raise NotImplementedError


# ------------------------------- OpenAI -------------------------------
class OpenAICaller(BaseCaller):
    """
    OpenAI Chat Completions 기반 생성형 분류기.

    특징:
    - 프롬프트로 **라벨만 출력**을 강하게 요구하고, 결과는 pick_label로 필터링.
    - 실패(인증/쿼터/네트워크)는 요소별 에러 문자열로 반환해 파이프라인 전체가 끊기지 않음.

    환경:
    - pip install openai
    - 환경변수 OPENAI_API_KEY 필요
    """

    def __call__(self, data: List[str]) -> List[str]:
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다.")
            client = OpenAI(api_key=api_key)
        except Exception as e:
            warnings.warn(f"OpenAI 호출 준비 실패: {e}")
            return ["[OpenAI 연결 실패]"] * len(data)

        outs: List[str] = []
        for text in data:
            prompt = self.adapter.generative_prompt(text)
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max(8, int(self.cfg.max_token)),
                )
                pred = (resp.choices[0].message.content or "").strip()
                outs.append(self.adapter.pick_label(pred))
            except Exception as e:
                outs.append(f"[OpenAI 오류: {e}]")
        return outs


# ---------------------------- Hugging Face ----------------------------
class HFZeroShotCaller(BaseCaller):
    """
    Hugging Face Transformers의 zero-shot-classification 파이프라인(NLI 기반).

    특징:
    - 생성형 대비 포맷 안정성 최고: 항상 후보 라벨 중 1개를 반환 → 후처리 불필요.
    - 다국어 NLI 체크포인트 사용(MNLI/XNLI 파인튜닝)으로 짧은 한국어 문장에도 강함.

    모델 전략:
    - 1차: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (경량/속도 유리)
    - 실패 시: joeddav/xlm-roberta-large-xnli (정확도↑/메모리·속도 비용↑)

    환경:
    - pip install transformers accelerate
    - GPU가 있으면 device=0, 없으면 CPU(-1)
    """

    def __call__(self, data: List[str]) -> List[str]:
        try:
            from transformers import pipeline
            from transformers.utils import logging
            logging.set_verbosity_error()
        except Exception as e:
            warnings.warn(f"HuggingFace import 실패: {e}")
            return ["[HF 연결 실패]"] * len(data)

        device = 0 if (_HAS_TORCH and torch.cuda.is_available()) else -1

        try:
            zsc = pipeline(
                task="zero-shot-classification",
                model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                device=device,
            )
        except Exception as e1:
            warnings.warn(f"기본 NLI 모델 준비 실패: {e1} → XLM-R large로 대체")
            try:
                zsc = pipeline(
                    task="zero-shot-classification",
                    model="joeddav/xlm-roberta-large-xnli",
                    device=device,
                )
            except Exception as e2:
                warnings.warn(f"대체 NLI 모델 준비 실패: {e2}")
                return ["[HF 연결 실패]"] * len(data)

        try:
            results = zsc(
                sequences=data,
                candidate_labels=self.adapter.labels,
                hypothesis_template=self.adapter.zshot_template(),
                multi_label=False,
                batch_size=16,
            )
        except Exception as e:
            return [f"[HF 오류: {e}]"] * len(data)

        if isinstance(results, dict):
            results = [results]
        return [(r.get("labels") or [""])[0] for r in results]


# -------------------------------- Gemini ------------------------------
class GeminiCaller(BaseCaller):
    """
    Google Generative AI(Gemini) 기반 생성형 분류기.

    특징:
    - OpenAI와 동일하게 프롬프트로 "라벨만"을 강력히 요구하고, pick_label로 후처리.
    - 안전성/정책 필터로 인해 간혹 장황 출력이 섞일 수 있어 후처리가 특히 중요.

    환경:
    - pip install google-generativeai
    - 환경변수 GEMINI_API_KEY (또는 GOOGLE_API_KEY) 필요

    개인적으로, gemini가 결과값이 제일 잘 나오는 것 같습니다.
    """

    def __call__(self, data: List[str]) -> List[str]:
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY 환경변수가 없습니다.")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            warnings.warn(f"Gemini 호출 준비 실패: {e}")
            return ["[Gemini 연결 실패]"] * len(data)

        outs: List[str] = []
        for text in data:
            prompt = self.adapter.generative_prompt(text)
            try:
                r = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.0,
                        "max_output_tokens": max(8, int(self.cfg.max_token)),
                        "top_p": 1.0,
                        "top_k": 1,
                    },
                )
                outs.append(self.adapter.pick_label(getattr(r, "text", "")))
            except Exception as e:
                outs.append(f"[Gemini 오류: {e}]")
        return outs
# -------------------------------- LLaMA -------------------------------
# 미완성
class LlamaCaller(BaseCaller):
    """
    LLaMA(및 호환 인스트럭트 LM) 기반 생성형 분류기.

    문제와 해결:
    - 생성형 모델은 후보 라벨을 모두 나열하거나 장황하게 답하는 경향이 있어,
      단순 '부분일치' 후처리만 쓰면 첫 라벨(예: "분노")로 쏠리는 바이어스가 발생한다..
      현재 이 부분을 수정하고 있으며, **숫자만 출력**하도록 프롬프트를 설계하고, 한 자리 숫자를 라벨로 매핑하는 방식을 통해
      바이어스를 좀 해결하려고 하고 있으나,, 아직 해결 못 한 상태.
      (max_new_tokens=2, temperature=0, return_full_text=False 등으로 강하게 제약)

    모델:
    - 기본: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (공개/경량/Colab CPU 가능)
    - 교체: 환경변수 HF_LLM_MODEL 또는 cfg.hf_model_name으로 HF 레포 id 지정
      (예: meta-llama/Meta-Llama-3.2-1B-Instruct — 접근 승인/토큰 필요할 수 있음)

    환경:
    - pip install transformers accelerate sentencepiece torch
    - GPU가 있으면 device_map="auto"로 자동 할당, 없으면 CPU에서도 동작(느릴 수 있음)
    """

    _digit_re = re.compile(r"\b([1-9])\b")

    def __init__(self, cfg: LLMConfig):
        super().__init__(cfg)
        self.model_name = (
            os.getenv("HF_LLM_MODEL")
            or getattr(cfg, "hf_model_name", None)
            or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        try:
            from transformers import AutoTokenizer, pipeline
            try:
                tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            except Exception:
                tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

            pipe_kwargs = {}
            if _HAS_TORCH:
                pipe_kwargs["device_map"] = "auto"
                pipe_kwargs["torch_dtype"] = "auto"

            self._tok = tok
            self._gen = pipeline("text-generation", model=self.model_name, tokenizer=tok, **pipe_kwargs)
        except Exception as e:
            warnings.warn(f"LLaMA 모델 로딩 실패: {e}")
            self._gen = None
            self._tok = None

        self.idx2label = {str(i + 1): lb for i, lb in enumerate(self.adapter.labels)}
        self.numbered = "; ".join([f"{i}) {lb}" for i, lb in self.idx2label.items()])

    def _build_prompt(self, text: str) -> str:
        """
        숫자만 출력하도록 강하게 유도하는 한국어 시스템 프롬프트를 구성.
        Chat 템플릿이 있는 토크나이저는 apply_chat_template로 역할(role)까지 반영.
        """

        instr = (
            f"다음 한국어 문장의 {self.adapter.task}를 번호로만 답하라.\n"
            f"라벨 후보: {self.numbered}\n"
            f"- 출력 형식: 오직 숫자 한 글자 (예: 1)\n"
            f"- 추가 문자, 공백, 설명 금지\n\n"
            f"문장: {text}\n정답 번호:"
        )
        tok = self._tok
        if tok and hasattr(tok, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "You are a strict classifier. Answer with ONLY the single index number."},
                {"role": "user", "content": instr},
            ]
            try:
                return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                return instr
        return instr

    def _parse_index(self, s: str) -> str:
        """
        생성된 텍스트에서 첫 한 자리 숫자를 추출하여 라벨로 변환.
        실패 시 빈 문자열 반환(외부에서 보수적 백업으로 문자열 라벨 매칭을 추가 수행).
        """

        if not s:
            return ""
        m = self._digit_re.search(s.strip())
        if not m:
            return ""
        return self.idx2label.get(m.group(1), "")

    def __call__(self, data: List[str]) -> List[str]:
        if self._gen is None:
            return ["[LLaMA 연결 실패]"] * len(data)

        outs: List[str] = []
        for text in data:
            prompt = self._build_prompt(text)
            try:
                y = self._gen(
                    prompt,
                    max_new_tokens=max(2, int(self.cfg.max_token)),
                    do_sample=False,
                    temperature=0.0,
                    return_full_text=False,
                    repetition_penalty=1.0,
                    eos_token_id=getattr(self._tok, "eos_token_id", None),
                )
                gen_text = (y[0].get("generated_text") or y[0].get("text") or "").strip()
                label = self._parse_index(gen_text)
                if not label:
                    label = self.adapter.pick_label(gen_text)
                outs.append(label)
            except Exception as e:
                outs.append(f"[LLaMA 오류: {e}]")
        return outs


# ---------------------------------------------------------------------
# 최상위 파이프라인
# ---------------------------------------------------------------------
class LLMPipeLine:
    """
    백엔드 선택과 호출을 담당하는 상위 파이프라인.

    역할:
    - 초기화 시 cfg.llm_name을 검증하고, 해당하는 Caller 인스턴스를 생성.
    - forward/__call__에서 입력 검증(List[str]) 후 Caller에게 위임.
    - 모든 Caller는 동일한 계약(입력/출력: List[str])을 지키므로 상위 레벨 코드가 단순해짐.

    사용 예:
        cfg = LLMConfig().CLASSIFY_SENTIMENT()
        cfg.llm_name = "gpt-opensrc"
        pipe = LLMPipeLine(cfg)
        labels = pipe(["나 슬퍼", "행복해"])
    """

    _DISPATCH: Dict[str, Callable[[LLMConfig], BaseCaller]] = {
        "gpt4o":       OpenAICaller, # OpenAI Chat Completions
        "gpt-opensrc": HFZeroShotCaller,  # Hugging Face zero-shot (NLI)
        "gemini":      GeminiCaller, # Google Generative AI
        #"llama":       LlamaCaller, # LLaMA / TinyLlama / 호환 모델
    }

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        llm = (cfg.llm_name or "").strip()
        if llm not in ALLOWED_LLM_NAMES:
            raise ValueError(f"지원하지 않는 llm_name='{llm}'. 허용값: {ALLOWED_LLM_NAMES}")
        self._caller = self._DISPATCH[llm](cfg)

    def forward(self, data: List[str]) -> List[str]:
        """
        런타임 입력 검증 후 Caller에 위임.
        - data: 반드시 List[str], 각 요소는 한 문장/텍스트
        - 반환: List[str], 각 요소는 최종 라벨 문자열(또는 에러 문자열)
        """

        if not isinstance(data, list) or (data and not isinstance(data[0], str)):
            raise ValueError("입력은 List[str]이어야 합니다.")
        return self._caller(data)

    def __call__(self, data: List[str]) -> List[str]:
        return self.forward(data)


# ---------------------------------------------------------------------
# 실행 예시
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 0) .env 로드 + 필요시 키 프롬프트 입력/저장
    load_api_keys("mindcast.env") #.env 파일 이름을 지정해뒀습니다. 
    # OpenAI/Gemini를 테스트할 때만 주석 해제하여 키를 안전하게 저장
    prompt_and_save_if_missing("OPENAI_API_KEY", "OpenAI API key")
    # prompt_and_save_if_missing("GEMINI_API_KEY", "Gemini API key")

    # 1) 감정 분류
    cfg_sent = LLMConfig().CLASSIFY_SENTIMENT()
    cfg_sent.llm_name = "gpt-opensrc" #모델 변
    cfg_sent.max_token = 10

    sent_data = ["나 오늘 너무 행복해", "짜증난다", "조금 불안해", "배가 아픈데 병원 가야 하나"]
    try:
        pipe = LLMPipeLine(cfg_sent)
        print("[SENT] 입력:", sent_data)
        print("[SENT] 출력:", pipe(sent_data))
    except (ValueError, NotImplementedError) as e:
        raise e

    # 2) 주제 분류
    cfg_topic = LLMConfig().CLASSIFY_TOPIC()
    cfg_topic.llm_name = "gpt-opensrc"
    cfg_topic.max_token = 10

    topic_data = ["우리나라 농구 개잘하네", "경제란 무엇일까", "재랑 재랑 연애한대ㄷㄷ"]
    try:
        pipe2 = LLMPipeLine(cfg_topic)
        print("[TOPIC] 입력:", topic_data)
        print("[TOPIC] 출력:", pipe2(topic_data))
    except (ValueError, NotImplementedError) as e:
        raise e

