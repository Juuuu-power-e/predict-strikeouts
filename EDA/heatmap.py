import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('../data/train.csv')
plt.rcParams['font.family'] = 'Malgun Gothic'
# 볼-스트라이크 조합별 삼진 확률
heatmap_data = train.groupby(['balls', 'strikes'])['k'].mean().unstack()
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('볼카운트에 따른 삼진 확률')
plt.xlabel('Strikes')
plt.ylabel('Balls')
plt.show()

pivot = train.pivot_table(values='k', index='outs_when_up', columns='n_thruorder_pitcher', aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('아웃 수 & 투수 순번별 삼진 확률')
plt.xlabel('n_thruorder_pitcher')
plt.ylabel('outs_when_up')
plt.show()


pivot = train.pivot_table(values='k', index='pitch_type', columns='stand', aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('투구 유형 & 타자 방향에 따른 삼진 확률')
plt.xlabel('타자 방향 (stand)')
plt.ylabel('투구 유형 (pitch_type)')
plt.show()


sns.jointplot(data=train, x='release_extension', y='arm_angle', kind='hex', gridsize=30, cmap='coolwarm')
plt.suptitle('릴리스 거리 vs 팔 각도 분포 (jointplot)', y=1.02)
plt.show()


sns.jointplot(data=train, x='release_extension', y='release_pos_z', kind='hex', gridsize=30, cmap='YlGnBu')
plt.suptitle('릴리스 위치 z vs 릴리스 거리 분포', y=1.02)
plt.show()
