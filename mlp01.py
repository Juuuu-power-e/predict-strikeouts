import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score
import os

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
SUBMIT_PATH = "submissions/submission_mlp01.csv"

class StrikeoutDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def check_tensor_nan(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"❌ {name} contains NaN!")
    if torch.isinf(tensor).any():
        print(f"❌ {name} contains Inf!")
    print(f"📊 {name} range: min={tensor.min().item():.5f}, max={tensor.max().item():.5f}")

def main():
    # 1. 데이터 로딩 및 전처리
    df = pd.read_csv(TRAIN_PATH).dropna()
    y = df["is_strike"]
    X_raw = df.drop(columns=["index", "is_strike"])
    X = pd.get_dummies(X_raw)
    feature_columns = X.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    train_ds = StrikeoutDataset(X_train, y_train.values)
    val_ds = StrikeoutDataset(X_val, y_val.values)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    # 2. 모델 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3. 학습 루프
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            check_tensor_nan(X_batch, "입력값 X_batch")

            optimizer.zero_grad()
            output = model(X_batch)

            check_tensor_nan(output, "출력값 output")

            loss = criterion(output, y_batch)

            if torch.isnan(loss):
                print("❌ Loss가 NaN입니다. 학습 중단")
                return

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f}")

    # 4. 검증 평가
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch).cpu().numpy().flatten()
            preds.extend(output)
            trues.extend(y_batch.numpy())

    print(f"✅ Validation LogLoss: {log_loss(trues, preds):.4f}")
    print(f"✅ Validation ROC AUC: {roc_auc_score(trues, preds):.4f}")

    # 5. 테스트셋 추론
    test_df = pd.read_csv(TEST_PATH)
    X_test_raw = test_df.drop(columns=["index"])
    X_test = pd.get_dummies(X_test_raw)

    # 누락된 컬럼 보정
    for col in feature_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_columns]

    X_test_scaled = scaler.transform(X_test)
    test_ds = StrikeoutDataset(X_test_scaled)
    test_loader = DataLoader(test_ds, batch_size=256)

    predictions = []
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch).cpu().numpy().flatten()
            predictions.extend(output)

    predictions = np.array(predictions)
    if np.isnan(predictions).any():
        print("⚠️ 예측값에 NaN이 포함되어 있습니다.")
    if np.isinf(predictions).any():
        print("⚠️ 예측값에 Inf가 포함되어 있습니다.")

    # NaN 값이 있으면 0.5로 대체 (or 다른 전략 사용 가능)
    predictions = np.nan_to_num(predictions, nan=0.5)

    # 6. 제출파일 저장
    os.makedirs("submissions", exist_ok=True)
    submit_df = pd.DataFrame({
        "index": test_df["index"].values,
        "k": predictions
    })
    submit_df.to_csv(SUBMIT_PATH, index=False)
    print(f"📁 Saved submission to: {SUBMIT_PATH}")

if __name__ == "__main__":
    main()
