import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터 불러오기
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# 2. A/B 그룹 분리
train_A = train_df[train_df["bat_speed"].notnull() & train_df["swing_length"].notnull()]
train_B = train_df[train_df["bat_speed"].isnull() & train_df["swing_length"].isnull()]
test_A = test_df[test_df["bat_speed"].notnull() & test_df["swing_length"].notnull()]
test_B = test_df[test_df["bat_speed"].isnull() & test_df["swing_length"].isnull()]

# 3. 주자 유무 파생 변수 생성
for df in [train_A, train_B, test_A, test_B]:
    df["has_runner"] = ((df["on_1b"] + df["on_2b"] + df["on_3b"]) > 0).astype(int)

# 4. 공통 전처리
drop_cols = ['index', 'on_1b', 'on_2b', 'on_3b', 'inning_topbot','inning']
cat_cols = ['pitch_type', 'pitch_name', 'stand', 'p_throws']

# ===== A 모델 (k 예측) =====
train_A = train_A.drop(columns=drop_cols + ['is_strike'])
test_A_idx = test_A['index']
test_A = test_A.drop(columns=drop_cols)

X_A = train_A.drop(columns=['k'])
y_A = train_A['k']

for col in cat_cols:
    if col in X_A.columns:
        X_A[col] = X_A[col].astype("category")
        test_A[col] = test_A[col].astype("category")

X_train_A, X_val_A, y_train_A, y_val_A = train_test_split(
    X_A, y_A, stratify=y_A, test_size=0.2, random_state=42
)

train_set_A = lgb.Dataset(X_train_A, y_train_A, categorical_feature=cat_cols)
val_set_A = lgb.Dataset(X_val_A, y_val_A, reference=train_set_A, categorical_feature=cat_cols)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.035,
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42
}

model_A = lgb.train(
    params,
    train_set_A,
    valid_sets=[train_set_A, val_set_A],
    num_boost_round=2000,
    callbacks=[early_stopping(50), log_evaluation(50)]
)

val_pred_A = model_A.predict(X_val_A)
print(f"[A 모델 LogLoss] val: {log_loss(y_val_A, val_pred_A):.5f}")
print(f"[A 모델 ROC AUC] val: {roc_auc_score(y_val_A, val_pred_A):.5f}")

test_pred_A = model_A.predict(test_A)

# ===== B 모델 (is_strike 예측 → 그대로 k로 사용) =====
train_B = train_B.drop(columns=drop_cols + ['k'])
test_B_idx = test_B['index']
test_B = test_B.drop(columns=drop_cols)

X_B = train_B.drop(columns=['is_strike'])
y_B = train_B['is_strike']

for col in cat_cols:
    if col in X_B.columns:
        X_B[col] = X_B[col].astype("category")
        test_B[col] = test_B[col].astype("category")

X_train_B, X_val_B, y_train_B, y_val_B = train_test_split(
    X_B, y_B, stratify=y_B, test_size=0.2, random_state=42
)

train_set_B = lgb.Dataset(X_train_B, y_train_B, categorical_feature=cat_cols)
val_set_B = lgb.Dataset(X_val_B, y_val_B, reference=train_set_B, categorical_feature=cat_cols)

params_b = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.025,
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42
}

model_B = lgb.train(
    params_b,
    train_set_B,
    valid_sets=[train_set_B, val_set_B],
    num_boost_round=3000,
    callbacks=[early_stopping(50), log_evaluation(50)]
)

val_pred_B = model_B.predict(X_val_B)
print(f"[B 모델 LogLoss] val: {log_loss(y_val_B, val_pred_B):.5f}")
print(f"[B 모델 ROC AUC] val: {roc_auc_score(y_val_B, val_pred_B):.5f}")

test_pred_B = model_B.predict(test_B)

# ===== 최종 결과 병합 및 저장 =====
sub_A = pd.DataFrame({'index': test_A_idx, 'k': test_pred_A})
sub_B = pd.DataFrame({'index': test_B_idx, 'k': test_pred_B})
final_sub = pd.concat([sub_A, sub_B]).sort_values("index")

save_path = "submissions/submission_split_lightgbm.csv"
base, ext = os.path.splitext(save_path)
i = 1
while os.path.exists(save_path):
    save_path = f"{base}({i}){ext}"
    i += 1

final_sub.to_csv(save_path, index=False)
print(f"✅ Submission saved to {save_path}")
