명령어(프로젝트 루트디렉토리에서 실행) :

python download_sarc_dataset.py



- SARC 2.0 Political Balanced 데이터를 자동으로 다운로드하고,  
    학습에 바로 사용할 수 있는 **`text`, `label`** 형태의 CSV 파일로 변환합니다.



- 커스텀 데이터셋 추가 형식
    새로운 데이터셋을 추가할 때에는 최종 저장되는 CSV 파일이 다음 형식을 따라야 합니다.

    | text                     | label  |
    | ------------------------ | ------ |
    | "[PARENT]: … [REPLY]: …" | 0 또는 1 |



- 컬럼 설명
    - text  
    `[PARENT]`, `[REPLY]` 로 구성된 대화 형태의 문자열이어야 함.
    예시:  
    ```
    [PARENT]: And we're upset since the Democrats would *never* try something as sneaky as this, right? [REPLY]: Oh yes, because honesty is their top priority.
    ```
    - label
    - `0`: 비풍자(non-sarcastic)  
    - `1`: 풍자(sarcastic)



-  요약
    | 작업 항목 | 실행 명령어 |
    |------------|--------------|
    | 🧾 데이터 다운로드 + 전처리 | `python download_sarc_dataset.py` |
    | 🧩 커스텀 데이터셋 추가 | `text`, `label` 컬럼으로 구성된 CSV 파일 작성 |
