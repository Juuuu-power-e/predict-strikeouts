import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

# 1. 데이터 로딩
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# 2. 결측치 처리
train_df = train_df.dropna(subset=['k']).fillna(0)
test_df = test_df.fillna(0)

# 3. 공통 수치형 feature만 사용
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
exclude_cols = ['k', 'is_strike']
common_cols = [col for col in numeric_cols if col in test_df.columns and col not in exclude_cols]

X = train_df[common_cols]
y = train_df['k']
test_X = test_df[common_cols]

# 4. 학습/검증 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 실험할 모델 정의
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    "LightGBM": lgb.LGBMClassifier()
}

results = []
trained_models = {}

# 6. 모델 훈련 및 평가
for name, model in models.items():
    print(f"\n🟢 Training {name}...")
    model.fit(X_train, y_train)
    val_pred = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, val_pred)
    print(f"✅ {name} ROC AUC: {score:.4f}")
    results.append((name, score))
    trained_models[name] = model

# 7. 결과 요약 출력
print("\n📊 Summary of Results:")
results.sort(key=lambda x: -x[1])
for name, score in results:
    print(f"{name:20}: ROC AUC = {score:.4f}")

# 8. 가장 성능 좋은 모델로 테스트셋 예측 및 제출 파일 생성
best_model_name = results[0][0]
best_model = trained_models[best_model_name]
print(f"\n📝 Generating submission with best model: {best_model_name}")

test_pred = best_model.predict_proba(test_X)[:, 1]
submission = pd.DataFrame({
    "index": test_df["index"],
    "k": test_pred
})
submission_filename = f"submission_{best_model_name.lower()}.csv"
submission.to_csv(submission_filename, index=False)
print(f"✅ Submission file saved as: {submission_filename}")
