import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# 경고 제거
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터 로드
df = pd.read_csv('data/train.csv')

# 2. 수치형 변수만 선택 (범주형 제외)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# 3. 상관계수 계산
corr_matrix = df[numeric_cols].corr()

# 4. 히트맵 시각화
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
plt.title('Feature Correlation Matrix', fontsize=18)
plt.tight_layout()
plt.show()


X = df[['arm_angle']]
y = df['release_extension']
model = LinearRegression().fit(X, y)
print(f"R² score: {r2_score(y, model.predict(X))}")
