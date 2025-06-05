import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
df = pd.read_csv('data/train.csv')  # 경로는 상황에 맞게 조정

# 2. 사용하지 않을 컬럼 제거
drop_cols = ['inning_topbot']  # 의미 없다고 판단한 피처
X = df.drop(columns=drop_cols + ['k'])  # 라벨은 'k'
y = df['k']

# categorical columns 처리
cat_cols = ['pitch_type', 'pitch_name', 'stand', 'p_throws']
for col in cat_cols:
    X[col] = X[col].astype('category')


# 4. 학습/검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. LightGBM Dataset으로 변환
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

# 6. 모델 파라미터 설정
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
        log_evaluation(period=50)  # 50마다 로그 찍음
    ]
)

# 8. 예측 및 평가
y_pred = model.predict(X_val)
print("📉 Log Loss:", log_loss(y_val, y_pred))
print("📈 ROC AUC:", roc_auc_score(y_val, y_pred))

# 9. 피처 중요도 시각화
lgb.plot_importance(model)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
