import pandas as pd

# 데이터 불러오기
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# 조건별 분리
train_A = train[train["bat_speed"].notnull() & train["swing_length"].notnull()]  # 스윙 있음
train_B = train[train["bat_speed"].isnull() & train["swing_length"].isnull()]    # 스윙 없음

test_A = test[test["bat_speed"].notnull() & test["swing_length"].notnull()]
test_B = test[test["bat_speed"].isnull() & test["swing_length"].isnull()]

# 저장 (선택사항)
train_A.to_csv("data/train_A.csv", index=False)
train_B.to_csv("data/train_B.csv", index=False)
test_A.to_csv("data/test_A.csv", index=False)
test_B.to_csv("data/test_B.csv", index=False)
