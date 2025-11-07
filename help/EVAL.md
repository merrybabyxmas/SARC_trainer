명령어(프로젝트 루트디렉토리에서 실행) :

python script/eval.py



- 산출물
eval_results/
├── eval_predictions.csv   ← (text, label, pred, prob)
├── eval_summary.txt       ← (accuracy, precision, recall, f1, confusion matrix)


결과 예시 : 

Accuracy: 0.8421
Precision: 0.8354
Recall: 0.8519
F1: 0.8436

Confusion Matrix:
[[504  89]
 [ 72 485]]
