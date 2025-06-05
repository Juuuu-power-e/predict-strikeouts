import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('data/train.csv')

# 주자 유무 파생 변수 생성
df['has_runner'] = ((df['on_1b'] + df['on_2b'] + df['on_3b']) > 0).astype(int)

# 수치형 컬럼만 선택
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# 상관관계 행렬 계산
corr_matrix = df[numeric_cols].corr()

# 히트맵 시각화
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
plt.title('Correlation Heatmap (with has_runner)', fontsize=18)
plt.tight_layout()
plt.show()
