import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import os

# 현재 파일 위치 기준으로 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, 'data', 'train.csv')

plt.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 데이터 로드
train = pd.read_csv(train_path)

# 결측치 시각화
missing = train.isnull().mean().sort_values(ascending=False)
missing = missing[missing > 0]

plt.figure(figsize=(12, 6))
sns.barplot(x=missing.index, y=missing.values, orient='v', width=0.3)
plt.title('결측치 비율이 있는 변수들')
plt.ylabel('결측 비율')
plt.xlabel('변수명')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# bat_speed와 swing_length 중 하나만 존재하는 행 확인
only_bat_speed = train['bat_speed'].notnull() & train['swing_length'].isnull()
only_swing_length = train['bat_speed'].isnull() & train['swing_length'].notnull()
one_only = train[only_bat_speed | only_swing_length]

print(f"\nbat_speed와 swing_length 중 하나만 존재하는 행 수: {len(one_only)}")
print(one_only[['bat_speed', 'swing_length']].head())
