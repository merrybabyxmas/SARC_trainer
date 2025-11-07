명령어(프로젝트 루트디렉토리에서 실행) :

python script/train.py



🧠 2️⃣ 모델 학습

학습 설정은 config/TRAINERCONFIG.py
에 정의되어 있습니다.
아래 명령어로 학습을 시작할 수 있습니다



| Setting       | Example                               |
| ------------- | ------------------------------------- |
| Base model    | `microsoft/deberta-v3-large`          |
| LoRA rank (r) | 8                                     |
| LoRA α        | 16                                    |
| Dropout       | 0.10                                  |
| Learning rate | 1e-4                                  |
| Batch size    | 32                                    |
| Epochs        | 300                                   |
| Device        | `cuda:0`                              |
| Train CSV     | `./data/sarc/train_bal_textlabel.csv` |
| Val CSV       | `./data/sarc/test_bal_textlabel.csv`  |


- 아직 다른 basemodel을 지원하지 않습니다. 
- 가급적 학습시 cuda:0말고, 1,2,3중에 사용하길 권장 
- train_csv, val_csv는 필요에 맞게 수정 


- 학습 출력물 

checkpoints_deltaWo_per_layer/
├── best.pt                 ← best model (highest F1)
├── training_curve.png      ← F1 / Accuracy / Loss curves
└── training_log.csv        ← per-epoch metrics