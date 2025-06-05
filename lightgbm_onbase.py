import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 데이터 불러오기
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 2. 주자 합산 파생 변수 생성
for df in [train, test]:
    df['on_base_sum'] = df[['on_1b', 'on_2b', 'on_3b']].sum(axis=1)

# 3. 사용하지 않을 컬럼 제거
drop_cols = ['index', 'on_1b', 'on_2b', 'on_3b', 'inning_topbot']
target = 'k'
train = train.drop(columns=drop_cols + ['is_strike'])  # train만 따로 처리
test_idx = test['index']
test = test.drop(columns=drop_cols)  # test에는 is_strike 없음

# 4. 특성과 타겟 분리
X = train.drop(columns=[target])
y = train[target]

# 5. 범주형 처리
cat_cols = ['pitch_type', 'pitch_name', 'stand', 'p_throws']
for col in cat_cols:
    if col in X.columns:
        X[col] = X[col].astype('category')
        test[col] = test[col].astype('category')

# 6. 학습/검증 분리
X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 7. LightGBM Dataset 생성
train_set = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)
val_set = lgb.Dataset(X_val, y_val, reference=train_set, categorical_feature=cat_cols)

# 8. 파라미터 및 모델 학습
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42
}

model = lgb.train(
    params,
    train_set,
    valid_sets=[train_set, val_set],
    num_boost_round=1000,
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=50)
    ]
)

# 9. 검증 성능 출력
val_pred = model.predict(X_val)
print(f"[LogLoss] val: {log_loss(y_val, val_pred):.5f}")
print(f"[ROC AUC] val: {roc_auc_score(y_val, val_pred):.5f}")

# 10. Feature Importance 시각화
lgb.plot_importance(model, max_num_features=20, importance_type='split', figsize=(8, 6))
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# 11. Test 예측 및 제출 파일 생성
test_pred = model.predict(test)
submission = pd.DataFrame({'index': test_idx, 'k': test_pred})

# 12. 파일 저장 (중복 방지)
save_path = "submissions/submission_lightgbm.csv"
base, ext = os.path.splitext(save_path)
i = 1
while os.path.exists(save_path):
    save_path = f"{base}({i}){ext}"
    i += 1

submission.to_csv(save_path, index=False)
print(f"✅ Submission saved to {save_path}")
