간단 명령어 

/LOCALPATH/mindcastlib/run/run_sequential_analysis.sh


1. 실행 설정 
    - 환경설정 : mindcastlib/configs/analysis_config.py 에서 확인 가능 
        DefaultConfig : SENT_CMT_TOPIC_TTL (감정분류 -> 댓글, 토픽분류 -> 타이틀, 반어법 ->댓글) 분석 수행
    
    - run/run_sequential_analysis.sh에서 
        INPUT_DIR : preprocessed 폴더 디렉토리 
        OUTPUT_DIR : 분석 결과 저장할 디렉토리 위치 
        설정한 후 실행



2. 입력 파일 구조 (preprocessed_data) : Sequential Analysis Pipeline은 preprocess 단계에서 생성된 데이터를 입력으로 사용 (1_PREPROCESS.md 참조) 

입력 폴더 구조:

preprocessed_data/
  ├── 2020/
  │   ├── 01/
  │   │   ├── 01-10/
  │   │   │   └── news_comments.json
  │   │   ├── 11-20/
  │   │   └── 21-31/


각 파일은 news_comments.json 형식.

3. 입력 데이터형식

아래는 sequential_analysis 가 필수적으로 요구하는 구조: (1_PREPROCESS.md 참조) 

{
  "data": [
    {
      "date": "YYYY-MM-DD",
      "posts": [
        {
          "title": "뉴스 제목",
          "raw_title": "파일명",
          "news_date": "YYYY-MM-DD",
          "comments": [
            "댓글1",
            "댓글2",
            "댓글3"
          ]
        }
      ]
    }
  ]
}

4. 분석 설명
    1. 종류 : 
        감정 분석 (Sentiment Classification), 
        토픽 분석 (Topic Classification)
        반어법 분석 (Sarcasm Detection)
    2. 적용 대상
        감정 분석 -> 댓글
        토픽 분석 -> 타이틀
        반어법 분석 -> 댓글


5. 📦 출력 데이터 구조 (analysis_results) : 출력 구조는 입력 구조와 동일한 구조로 저장됨.

analysis_results/
  ├── 2020/
  │   ├── 01/
  │   │   ├── 01-10/
  │   │   │   └── infer_20250110_153012.json
  │   │   ├── 11-20/
  │   │   └── 21-31/


파일은 timestamp를 포함한 이름으로 저장됨:

infer_YYYYMMDD_HHMMSS.json

6. 🧾 출력 JSON 양식 (analysis result format)

아래는 결과 파일 내 구조 예시

{
  "data": [
    {
      "date": "2020-01-10",
      "posts": [
        {
          "title": "뉴스제목",
          "raw_title": "뉴스제목",
          "news_date": "2020-01-01",
          "comments": [
            "댓글1",
            "댓글2"
          ],
          "analyses": {
            "SarcasmDetectionPipeLine_comments": [
              [
                {
                  "label": "sarcastic or non-sarcastic",
                  "score": 0.xx
                }
              ]
            ],
            "SentimentClassificationPipeLine_comments": [
              [
                {
                  "label": "분노/슬픔/기쁨/불안/상처/당황",
                  "score": 0.xx
                }
              ]
            ],
            "text-classification_title": [
              {
                "label": "정치/사회/경제/문화/국제/IT 등",
                "score": 0.xx
              }
            ]
          }
        }
      ]
    }
  ]
}



| 필드명                                        | 설명                       
| ------------------------------------------ | ------------------------ 
| `SarcasmDetectionPipeLine_comments`        | 댓글별 반어 여부 분석 결과          
| `SentimentClassificationPipeLine_comments` | 댓글별 감정 분석 결과             
| `text-classification_title`                | 뉴스 제목의 토픽 분류 결과          
| `label`                                    | 예측 라벨                    
| `score`                                    | softmax confidence score (모델의 확신정도)