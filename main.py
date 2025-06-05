import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

# 1. 데이터 로딩
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# 2. 사용할 피처 정의
features = [
    'pitch_type', 'pitch_name', 'outs_when_up', 'balls', 'strikes',
    'n_thruorder_pitcher', 'stand', 'p_throws', 'sz_top', 'sz_bot',
    'pfx_x', 'pfx_z', 'arm_angle', 'release_speed', 'release_pos_x',
    'release_extension', 'release_pos_z', 'release_spin_rate', 'spin_axis',
    'bat_speed', 'swing_length'
]

# 3. 결측치 처리
train_df = train_df.dropna(subset=['k']).fillna(0)
test_df = test_df.fillna(0)

# 4. 범주형 변수 라벨 인코딩
categorical_cols = train_df[features].select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])  # 테스트셋은 transform만 해야 함
    label_encoders[col] = le

# 5. 데이터 분리
X = train_df[features]
y = train_df['k']
test_X = test_df[features]

# 6. 학습/검증 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 7. 실험할 모델 정의
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    "LightGBM": lgb.LGBMClassifier()
}

results = []
trained_models = {}

# 8. 모델 훈련 및 평가
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    val_pred = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, val_pred)
    print(f"{name} ROC AUC: {score:.4f}")
    results.append((name, score))
    trained_models[name] = model

# 9. 결과 요약 출력
print("\nSummary of Results:")
results.sort(key=lambda x: -x[1])
for name, score in results:
    print(f"{name:20}: ROC AUC = {score:.4f}")

# 10. 가장 성능 좋은 모델로 테스트셋 예측 및 제출 파일 생성
best_model_name = results[0][0]
best_model = trained_models[best_model_name]
print(f"\nGenerating submission with best model: {best_model_name}")

test_pred = best_model.predict_proba(test_X)[:, 1]
submission = pd.DataFrame({
    "index": test_df["index"],
    "k": test_pred
})

# 파일 저장 경로 설정
import os
submission_dir = "submissions"
os.makedirs(submission_dir, exist_ok=True)

filename = f"submission_{best_model_name.lower()}.csv"
filepath = os.path.join(submission_dir, filename)

# 중복 방지 파일명 설정
count = 1
while os.path.exists(filepath):
    filepath = os.path.join(submission_dir, f"submission_{best_model_name.lower()}({count}).csv")
    count += 1

submission.to_csv(filepath, index=False)
print(f"✅ Submission file saved as: {filepath}")
