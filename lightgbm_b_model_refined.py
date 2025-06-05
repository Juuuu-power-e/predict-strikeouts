import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import os

# 1. 데이터 불러오기
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# 2. B 그룹 추출 (스윙 없음)
train_B = train_df[train_df["bat_speed"].isnull() & train_df["swing_length"].isnull()]
test_B = test_df[test_df["bat_speed"].isnull() & test_df["swing_length"].isnull()]
test_B_idx = test_B['index']

# 3. 유의미한 피처만 선택
useful_features = [
    'pitch_type', 'pitch_name', 'stand', 'p_throws',
    'sz_top', 'sz_bot', 'arm_angle',
    'release_pos_x', 'release_pos_z', 'release_speed',
    'pfx_x', 'pfx_z', 'release_extension'
]

train_B = train_B[useful_features + ['is_strike']]
test_B = test_B[useful_features]

# 4. 범주형 처리
cat_cols = ['pitch_type', 'pitch_name', 'stand', 'p_throws']
for col in cat_cols:
    train_B[col] = train_B[col].astype("category")
    test_B[col] = test_B[col].astype("category")

# 5. train/val split
X = train_B.drop(columns=['is_strike'])
y = train_B['is_strike']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 6. LightGBM Dataset
train_set = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)
val_set = lgb.Dataset(X_val, y_val, reference=train_set, categorical_feature=cat_cols)

# 7. 파라미터 설정
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42
}

# 8. 학습
model = lgb.train(
    params,
    train_set,
    valid_sets=[train_set, val_set],
    num_boost_round=1000,
    callbacks=[early_stopping(50), log_evaluation(50)]
)

# 9. 성능 출력
val_pred = model.predict(X_val)
print(f"[개선 B 모델 LogLoss] val: {log_loss(y_val, val_pred):.5f}")
print(f"[개선 B 모델 ROC AUC] val: {roc_auc_score(y_val, val_pred):.5f}")

# 10. 예측 수행 (확률 그대로 저장)
test_pred = model.predict(test_B)

# 11. 제출 파일 저장
submission = pd.DataFrame({'index': test_B_idx, 'k': test_pred})
save_path = "submissions/submission_B_refined.csv"
base, ext = os.path.splitext(save_path)
i = 1
while os.path.exists(save_path):
    save_path = f"{base}({i}){ext}"
    i += 1

submission.to_csv(save_path, index=False)
print(f"✅ 개선된 B 모델 결과 저장됨: {save_path}")
