import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

train = pd.read_csv('../data/train.csv')
plt.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지
# 수치형 변수 추출
numeric_cols = train.select_dtypes(include=['float64', 'int64']).drop(columns=['index', 'k']).columns.tolist()

# 시각화
n_cols = 3
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
plt.figure(figsize=(n_cols * 5, n_rows * 3.5))

for idx, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, idx)
    sns.histplot(train[col], bins=30, kde=True)
    plt.title(col)

plt.tight_layout()
plt.show()
