import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import lightgbm as lgb
import os

# 데이터 로딩
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# 피처 정의
features = [
    'pitch_type', 'pitch_name', 'outs_when_up', 'balls', 'strikes',
    'n_thruorder_pitcher', 'stand', 'p_throws', 'sz_top', 'sz_bot',
    'pfx_x', 'pfx_z', 'arm_angle', 'release_speed', 'release_pos_x',
    'release_extension', 'release_pos_z', 'release_spin_rate', 'spin_axis',
    'bat_speed', 'swing_length'
]

# 결측치 처리
train_df = train_df.dropna(subset=['k']).fillna(0)
test_df = test_df.fillna(0)

# 공통 분리
X_all = train_df[features]
y_all = train_df['k']
test_X_all = test_df[features]

# 두 가지 인코딩 결과 저장용
encoding_results = {}

# ---------------------------
# 1. Label Encoding 방식
# ---------------------------
train_le = train_df.copy()
test_le = test_df.copy()

categorical_cols = train_le[features].select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_le[col] = le.fit_transform(train_le[col])
    test_le[col] = le.transform(test_le[col])
    label_encoders[col] = le

X = train_le[features]
y = train_le['k']
test_X = test_le[features]

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

models = {
    "LightGBM": lgb.LGBMClassifier(),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

label_encoding_results = []
print("\n🔷 Label Encoding Results")
for name, model in models.items():
    model.fit(X_train, y_train)
    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    loss = log_loss(y_val, val_pred)
    print(f"{name:20}: ROC AUC = {auc:.4f} | LogLoss = {loss:.4f}")
    label_encoding_results.append((name, auc, loss))

encoding_results["Label Encoding"] = label_encoding_results

# ---------------------------
# 2. One-Hot Encoding 방식
# ---------------------------
print("\n🔷 One-Hot Encoding Results")
categorical_cols = X_all.select_dtypes(include='object').columns.tolist()
numeric_cols = [col for col in features if col not in categorical_cols]

# 전처리 파이프라인 구성
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ct = ColumnTransformer(transformers=[
    ('cat', ohe, categorical_cols)
], remainder='passthrough')

X_encoded = ct.fit_transform(X_all)
test_X_encoded = ct.transform(test_X_all)

X_train, X_val, y_train, y_val = train_test_split(X_encoded, y_all, stratify=y_all, test_size=0.2, random_state=42)

onehot_encoding_results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    loss = log_loss(y_val, val_pred)
    print(f"{name:20}: ROC AUC = {auc:.4f} | LogLoss = {loss:.4f}")
    onehot_encoding_results.append((name, auc, loss))

encoding_results["One-Hot Encoding"] = onehot_encoding_results

# ---------------------------
# 결과 요약 출력
# ---------------------------
print("\n📊 Summary of Results:")
for enc_type, results in encoding_results.items():
    print(f"\n💠 {enc_type}")
    results.sort(key=lambda x: x[2])  # LogLoss 기준 정렬
    for name, auc, loss in results:
        print(f"{name:20}: ROC AUC = {auc:.4f} | LogLoss = {loss:.4f}")
