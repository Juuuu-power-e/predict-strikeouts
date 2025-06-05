import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import os

# 1. 데이터 불러오기
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# 2. 주자 유무 파생 변수 생성
for df in [train_df, test_df]:
    df["has_runner"] = ((df["on_1b"] + df["on_2b"] + df["on_3b"]) > 0).astype(int)

# 3. 공통 전처리
drop_cols = ['index', 'on_1b', 'on_2b', 'on_3b', 'inning_topbot', 'inning', 'is_strike']
cat_cols = ['pitch_type', 'pitch_name', 'stand', 'p_throws']

train_df = train_df.drop(columns=drop_cols)
test_idx = test_df['index']
test_df = test_df.drop(columns=drop_cols[:-1])  # test에는 is_strike 없음

X = train_df.drop(columns=['k'])
y = train_df['k']

# 범주형 처리
for col in cat_cols:
    if col in X.columns:
        X[col] = X[col].astype("category")
        test_df[col] = test_df[col].astype("category")

# 4. 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 5. LightGBM Dataset 생성
train_set = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)
val_set = lgb.Dataset(X_val, y_val, reference=train_set, categorical_feature=cat_cols)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.03,   # 더 작게 조정
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42
}


model = lgb.train(
    params,
    train_set,
    valid_sets=[train_set, val_set],
    num_boost_round=2000,
    callbacks=[early_stopping(50), log_evaluation(50)]
)

# 6. 검증 성능
val_pred = model.predict(X_val)
print(f"[전체 A 방식 모델 LogLoss] val: {log_loss(y_val, val_pred):.5f}")
print(f"[전체 A 방식 모델 ROC AUC] val: {roc_auc_score(y_val, val_pred):.5f}")

# 7. 테스트 예측
test_pred = model.predict(test_df)

# 8. 제출 파일 생성
submission = pd.DataFrame({'index': test_idx, 'k': test_pred})

save_path = "submissions/submission_allA_lightgbm.csv"
base, ext = os.path.splitext(save_path)
i = 1
while os.path.exists(save_path):
    save_path = f"{base}({i}){ext}"
    i += 1

submission.to_csv(save_path, index=False)
print(f"✅ Submission saved to {save_path}")
