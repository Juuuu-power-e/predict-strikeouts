import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import matplotlib.pyplot as plt
from lightgbm import early_stopping, log_evaluation
import os

# 1. 데이터 불러오기
df = pd.read_csv('data/train.csv')

# 🛠️ index 열이 컬럼으로 들어와 있다면 제거
if 'index' in df.columns:
    df = df.drop(columns=['index'])

# 2. 사용하지 않을 컬럼 제거
drop_cols = ['inning_topbot', 'is_strike', 'on_1b', 'on_2b', 'on_3b', 'outs_when_up']
X = df.drop(columns=drop_cols + ['k'])
y = df['k']

# 3. 범주형 변수 처리
cat_cols = ['pitch_type', 'pitch_name', 'stand', 'p_throws']
for col in cat_cols:
    if col in X.columns:
        X[col] = X[col].astype('category')

# ✅ train에 사용된 feature 목록 저장
used_features = X.columns.tolist()

# 4. 학습/검증 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 5. LightGBM Dataset 생성
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

# 6. LightGBM 파라미터 설정
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'seed': 42
}

# 7. 모델 학습
model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    callbacks=[
        early_stopping(stopping_rounds=20),
        log_evaluation(period=50)
    ]
)

# 8. 검증 성능 출력
y_pred_val = model.predict(X_val)
print("📉 Log Loss:", log_loss(y_val, y_pred_val))
print("📈 ROC AUC:", roc_auc_score(y_val, y_pred_val))

# 9. 피처 중요도 시각화
lgb.plot_importance(model)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# 10. test 데이터 불러오기 (index가 id임)
test = pd.read_csv('data/test.csv', index_col=0)

# 11. train과 동일한 feature만 선택
X_test = test[used_features].copy()

# 12. 범주형 처리
for col in cat_cols:
    if col in X_test.columns:
        X_test[col] = X_test[col].astype('category')

# 13. 예측 수행
test_preds = model.predict(X_test)

# 14. 제출 형식 변환
submission = pd.DataFrame({
    'index': X_test.index,
    'k': test_preds
})

# 15. 제출 파일 저장 (중복 방지)
base_path = 'submissions/submission_lightgbm.csv'
file_path = base_path
counter = 1
while os.path.exists(file_path):
    file_path = base_path.replace('.csv', f'({counter}).csv')
    counter += 1

submission.to_csv(file_path, index=False)
print(f"✅ 제출 파일 저장 완료: {file_path}")
