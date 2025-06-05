import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from lightgbm import early_stopping
from datetime import datetime

# 1. 데이터 불러오기
train_path = os.path.join('data', 'train.csv')
test_path = os.path.join('data', 'test.csv')
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 2. 사용할 feature 목록
selected_features = [
    'pitch_type', 'pitch_name',
    'on_1b', 'on_2b', 'on_3b',
    'outs_when_up', 'balls', 'strikes',
    'n_thruorder_pitcher', 'stand', 'p_throws',
    'sz_top', 'sz_bot', 'pfx_x', 'pfx_z',
    'release_extension', 'release_spin_rate', 'spin_axis',
    'bat_speed', 'swing_length'
]

# 3. X, y 분리
X = train_df[selected_features].copy()
y = train_df['k']

# 4. 결측치 처리 (범주형: 'Unknown', 수치형: 0)
for col in X.columns:
    if X[col].dtype.name == 'category' or X[col].dtype == object:
        X[col] = X[col].astype('category')
        X[col] = X[col].cat.add_categories(['Unknown'])
        X[col] = X[col].fillna('Unknown')
    else:
        X[col] = X[col].fillna(0)

# 5. 학습/검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 6. LightGBM Dataset 생성
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature='auto')
val_data = lgb.Dataset(X_val, label=y_val, categorical_feature='auto')

# 7. 모델 학습
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'verbose': -1,
    'random_state': 42
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=1000,
    callbacks=[early_stopping(50)]
)

# 8. 검증 성능 출력
val_pred = model.predict(X_val)
print(f"✅ Log Loss: {log_loss(y_val, val_pred):.4f}")
print(f"✅ ROC AUC : {roc_auc_score(y_val, val_pred):.4f}")

# 9. test.csv 처리
X_test = test_df[selected_features].copy()

for col in X_test.columns:
    if X_test[col].dtype.name == 'category' or X_test[col].dtype == object:
        X_test[col] = X_test[col].astype('category')
        X_test[col] = X_test[col].cat.add_categories(['Unknown'])
        X_test[col] = X_test[col].fillna('Unknown')
    else:
        X_test[col] = X_test[col].fillna(0)

# 10. 예측
test_pred = model.predict(X_test)

# 11. 제출 파일 생성
submission = pd.DataFrame({
    'index': test_df['index'],
    'k': test_pred
})

# 파일명 중복 방지
base_name = 'submission_lightgbm.csv'
output_dir = 'submissions'
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, base_name)
i = 1
while os.path.exists(output_path):
    base_name = f'submission_lightgbm({i}).csv'
    output_path = os.path.join(output_dir, base_name)
    i += 1

submission.to_csv(output_path, index=False)
print(f"✅ 제출 파일 저장 완료: {output_path}")
