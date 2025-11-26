간단 명령어 

/LOCALPATH/mindcastlib/run/run_single_analysis.sh


sequential_analysis는 폴더 내의 모든 json파일 순회하며 분석 수행하는 반면, 개별 json파일을 지정해서 분석하는 기능 


1. 실행 설정 
    - 환경설정 : mindcastlib/configs/analysis_config.py 에서 확인 가능 
        DefaultConfig : SENT_CMT_TOPIC_TTL (감정분류 -> 댓글, 토픽분류 -> 타이틀, 반어법 ->댓글) 분석 수행
    
    - run/run_sequential_analysis.sh에서 
        INPUT_DIR : preprocessed json 파일 위치
        OUTPUT_DIR : 분석 결과 저장할 파일 위치 
        설정한 후 실행


입력 데이터 : json file
출력 데이터 : json file 

이외에는 sequential analysis와 동일(2_0_ANALYSIS.md 참조)