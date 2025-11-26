간단 명령어 

/LOCALPATH/mindcastlib/run/run_preprocess.sh




1. preprocess 실행 설정 : 뉴스 JSON 파일을 날짜별/게시일별로 정제하여 분석 가능한 형태로 변환
    - 환경설정 : mindcastlib/configs/preprocess_config.py 에서 확인 가능 
        DefaultConfig : 원본 뉴스데이터에서 ["title", "raw_title", "news_date", "comments"]의 정보만을 가지고 전처리
        
    - run/run_preprocss.sh에서 
        INPUT_DIR : 원본 입력데이터 폴더 디렉토리 
        OUTPUT_DIR : 전처리 데이터 저장할 디렉토리 위치 
        설정한 후 실행




2. 입력 파일 저장 구조 : 아래와 같은 폴더 구조를 기준으로 작동함.
original_data/
  ├── 2020/
  │   ├── 01/
  │   │   ├── 01-10/
  │   │   │   └── 뉴스1.json
  │   │   ├── 11-20/
  │   │   └── 21-31/




3. 입력 데이터 양식 : 아래와 같은 json 파일을 기준으로 작동함
<EX. 뉴스1.json>
[
    {
        "date": "YYYY-MM-DD"
    },
    {
        "kind": "youtube#commentThread",
        "etag": "...",
        "id": "...",
        "snippet": {
            "channelId": "...",
            "videoId": "...",
            "topLevelComment": {
                "kind": "youtube#comment",
                "etag": "...",
                "id": "...",
                "snippet": {
                    "channelId": "...",
                    "videoId": "...",
                    "textDisplay": "댓글 내용",
                    "textOriginal": "댓글 원본 내용",
                    "authorDisplayName": "...",
                    "authorProfileImageUrl": "...",
                    "authorChannelUrl": "...",
                    "authorChannelId": {
                        "value": "..."
                    },
                    "canRate": true,
                    "viewerRating": "none",
                    "likeCount": 0,
                    "publishedAt": "YYYY-MM-DDTHH:MM:SSZ",
                    "updatedAt": "YYYY-MM-DDTHH:MM:SSZ"
                }
            },
            "canReply": true,
            "totalReplyCount": 0,
            "isPublic": true
        }
    },
    {
        "... 추가 댓글 ..."
    }
]


3. 출력 데이터 구조 : 전처리 후 결과는 입력과 동일한 폴더 구조로 저장됨:

preprocessed_data/
  ├── 2020/
  │   ├── 01/
  │   │   ├── 01-10/
  │   │   │   └── news_comments.json
  │   │   ├── 11-20/
  │   │   └── 21-31/




4. 출력 데이터 형식 : 출력되는 데이터는 아래와 같은 정보들을 포함하게 됌. 
출력 JSON 형식 (news_comments.json)
{
    "data": [
        {
            "date": "2020-01-14",
            "posts": [
                {
                    "title": "뉴스제목",
                    "raw_title": "뉴스제목",
                    "news_date": "2020-01-06",
                    "comments": ["댓글1", "댓글2", "댓글3"]
                }
            ]
        }
    ]
}

출력 기준

댓글의 publishedAt 날짜 기준으로 묶임

같은 날짜의 여러 뉴스도 posts 안에 병합

JSON 파일명은 항상 news_comments.json