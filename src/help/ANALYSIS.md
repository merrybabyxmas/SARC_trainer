정신건강 관련 분석을 위한 패키지 


간단 명령어 :  python -m mindcastlib.scripts.sequential_analysis (프로젝트 상위 루트에서 실행)



1. analysis : 뉴스의 제목과 댓글을 대상으로 감정분석, 토픽분석, 반어법분석 을 수행 

    - 환경설정 : /home/dongwoo38/mindcastlib/configs/model_config.py 에서 확인 가능 
        1. 감정분석 : klue-bert-base-sentiment 사용 (전체 60분류 6분류로 합쳐서 사용)
        2. 토픽분석 : freud-sensei/headline_classification 사용 
        3. 반어분석 : base-model : deberta-v3-large(한국어 모델로 대체 필요) + fine-tuned layer = assets/sarc.pt (우리 모델)
        4. SENT_CMT_TOPIC_TTL(디폴트 config) : 감정분석==>COMMENT, 토픽분석==>TITLE, 반어법(SARCASM)==>COMMENTS 대상으로 분석. 그 외 조합도 
            (ex. 타이틀에 감정분석) 커스텀 가능. 커스텀하면 mindcastlib/scripts/sequential_analysis.py의 run_analysis_pipeline의 
                runner = AnalysisPipeLine(
                analysis_config=AnalysisConfig.SENT_CMT_TOPIC_TTL(),) 부분의 config를 custom config로 바꿔야함. 

    - 입력 데이터 양식 : 아래 형식의 json 형식을 입력해야함. 
        {
        "data": [
            {
            "date": "날짜",
            "posts": [
                {
                "title": "제목1",
                "raw_title": "제목1(title과 동일)",
                "news_date": "날짜",
                "comments": ["댓글1", "댓글2" ,,,"댓글N"]
                }
            ],
            [
                {
                "title": "제목2",
                "raw_title": "제목2(title과 동일)",
                "news_date": "날짜",
                "comments": ["댓글1", "댓글2" ,,,"댓글N"]
                }
            ],
            }
        ]
        }
        *** 아래의 json파일을 날짜별 폴더 형식으로 저장
        *** EX. data/2020/02/01-10/news_comments.json, 2020/02/11-20/news_comments.json,,,
    

    - 명령어 : 프로젝트 상위루트에서 실행해야 함. 
        
        python -m mindcastlib.scripts.sequential_analysis
