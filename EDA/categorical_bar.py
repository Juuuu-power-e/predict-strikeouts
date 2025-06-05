import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# 📌 상대 경로 기준으로 데이터 불러오기
train = pd.read_csv('../data/train.csv')
plt.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지
# 범주형 변수 정의
cat_cols = train.select_dtypes(include='object').columns.tolist() + ['inning_topbot', 'stand', 'p_throws']

# 범주형 변수 분포 시각화
for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=train, order=train[col].value_counts().index)
    plt.title(f'{col} 분포')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# 타겟(k)에 따른 범주별 삼진 확률 시각화
for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.barplot(x=col, y='k', data=train, estimator='mean', order=train[col].value_counts().index)
    plt.title(f'{col}별 삼진 확률')
    plt.ylabel('k (삼진 확률)')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
