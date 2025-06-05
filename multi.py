import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

# 데이터 로딩
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# 전처리
train_df = train_df.dropna(subset=['k']).fillna(0)
test_df = test_df.fillna(0)

# 공통 수치형 피처 선택
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
exclude_cols = ['k', 'is_strike']
common_cols = [col for col in numeric_cols if col in test_df.columns and col not in exclude_cols]

X = train_df[common_cols]
y = train_df['k']
test_X = test_df[common_cols]

# 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 실험할 모델 목록
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": lgb.LGBMClassifier()
}

results = []

# 모델 반복 학습
for name, model in models.items():
    print(f"\n🟢 Training {name}...")
    model.fit(X_train, y_train)
    val_pred = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, val_pred)
    print(f"✅ {name} ROC AUC: {score:.4f}")
    results.append((name, score))

# 결과 출력
print("\n📊 Summary of Results:")
for name, score in sorted(results, key=lambda x: -x[1]):
    print(f"{name:20}: ROC AUC = {score:.4f}")
