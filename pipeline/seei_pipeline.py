import json
import math
import pandas as pd
from datetime import datetime
from collections import defaultdict
import plotly.graph_objects as go


# ======================================================
# 🔧 1) 외부 JSON 설정 파일 불러오기
# ======================================================
def load_config(json_path="/home/yein40/mindcastlib/configs/suicide/2022_06.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ======================================================
# 🔧 2) 부정 감정 세트
# ======================================================
NEGATIVE_SET = ["분노", "슬픔", "불안", "상처", "당황"]


# ======================================================
# 🔧 3) Helper Functions
# ======================================================
def extract_negative_ratio(sentiments):
    if len(sentiments) == 0:
        return 0
    neg = sum(1 for s in sentiments if s in NEGATIVE_SET)
    return neg / len(sentiments)


def exposure_intensity(num_comments):
    return math.log(num_comments + 1)


def direction_of(neg_ratio):
    return 1 if neg_ratio >= 0.5 else -1


def detect_period(date_str):
    day = int(date_str.split("-")[2])
    if day <= 10:
        return 1
    elif day <= 20:
        return 2
    else:
        return 3


# ======================================================
# 🔧 4) 이 부분에서 네 config를 이용해 카테고리 매핑
# ======================================================
def detect_category(title, suicide_keyword_mask, suicide_subtag_mask, config):
    matched = []

    # suicide keyword mask 기반 1차 분류
    for cat, flag in suicide_keyword_mask.items():
        if flag and cat in config["keywords"]:
            matched.append(cat)

    # suicide subtag mask 기반 2차 분류
    for subtag, flag in suicide_subtag_mask.items():
        if not flag:
            continue
        for cat, kw_list in config["keywords"].items():
            if subtag in kw_list:
                matched.append(cat)

    # 타이틀 직접 키워드 매칭 (백업)
    for cat, kw_list in config["keywords"].items():
        for kw in kw_list:
            if kw in title:
                matched.append(cat)

    return list(set(matched))


# ======================================================
# 🔧 5) Main SEEI Pipeline
# ======================================================
def run_seii(json_path, config_path="/home/yein40/mindcastlib/configs/suicide/2022_06.json"):
    config = load_config(config_path)

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    news_rows = []
    period_sum = defaultdict(lambda: defaultdict(float))

    for block in raw["data"]:
        for post in block["posts"]:

            title = post["title"]
            date = post["news_date"]
            comments = post.get("comments", [])

            # 댓글 감정 추출
            sentiments = [
                item[0]["label"]
                for item in post["analyses"].get("SentimentClassificationPipeLine_comments", [])
            ]

            neg_ratio = extract_negative_ratio(sentiments)
            exposure = exposure_intensity(len(sentiments))
            direct = direction_of(neg_ratio)
            seii_value = exposure * neg_ratio * direct

            suicide_res = post["analyses"]["SuicideDetectionPipeLine_title"][0]

            matched = detect_category(
                title,
                suicide_res["suicide_keyword_mask"],
                suicide_res["suicide_subtag_mask"],
                config
            )

            period = detect_period(date)

            # 카테고리별 합산
            for cat in matched:
                period_sum[period][cat] += seii_value

            news_rows.append({
                "date": date,
                "period": period,
                "title": title,
                "category": ", ".join(matched),
                "negative_ratio": neg_ratio,
                "exposure": exposure,
                "direction": direct,
                "SEEI": seii_value,
            })

    # 뉴스 단위 CSV
    df_news = pd.DataFrame(news_rows)
    df_news.to_csv("seei_news.csv", index=False)

    # 구간 단위 CSV
    period_rows = []
    for p in sorted(period_sum.keys()):
        rec = {"period": p}
        rec.update(period_sum[p])
        rec["SEEI_total"] = sum(period_sum[p].values())
        period_rows.append(rec)

    df_period = pd.DataFrame(period_rows)
    df_period.to_csv("seei_periods.csv", index=False)

    return df_news, df_period


# ======================================================
# 🔧 6) Dashboard
# ======================================================
def show_dashboard(df_period):
    fig = go.Figure()

    cats = [c for c in df_period.columns if c not in ["period", "SEEI_total"]]

    for cat in cats:
        fig.add_trace(go.Scatter(
            x=df_period["period"],
            y=df_period[cat],
            name=cat,
            mode="lines+markers"
        ))

    fig.add_trace(go.Scatter(
        x=df_period["period"],
        y=df_period["SEEI_total"],
        name="SEEI_total",
        mode="lines+markers",
        line=dict(width=4, color="red")
    ))

    fig.update_layout(
        title="10일 단위 외적 스트레스 지수(SEE Index)",
        xaxis_title="기간 (1=1~10일, 2=11~20일, 3=21~말일)",
        yaxis_title="SEEI 합산",
        template="plotly_white"
    )

    fig.show()


# ======================================================
# 🔧 7) 실행
# ======================================================
if __name__ == "__main__":
    df_news, df_period = run_seii("example_infer.json", config_path="seei_keywords.json")
    show_dashboard(df_period)
