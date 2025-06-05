# mlp03.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score
import os
import logging

# 로그 설정
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/mlp03.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
SUBMIT_PATH = "submissions/submission_mlp03.csv"
NUMERIC_CLIP_LIMIT = 10  # Z-score 클리핑 임계값

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
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def clip_numeric_features(df):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].clip(-NUMERIC_CLIP_LIMIT, NUMERIC_CLIP_LIMIT)
    return df

def main():
    df = pd.read_csv(TRAIN_PATH).dropna()
    y = df["is_strike"]
    X_raw = df.drop(columns=["index", "is_strike"])
    X_raw = clip_numeric_features(X_raw)
    X = pd.get_dummies(X_raw)
    feature_columns = X.columns

    # 이상치 의심 피처 출력
    high_value_cols = X.columns[(np.abs(X) > 100).any()]
    logging.warning(f"⚠️ 이상치 의심 피처: {list(high_value_cols)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    train_ds = StrikeoutDataset(X_train, y_train.values)
    val_ds = StrikeoutDataset(X_val, y_val.values)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=X.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            output = model(X_batch)

            loss = criterion(output, y_batch)
            if torch.isnan(loss):
                logging.error("❌ Loss is NaN. Training stopped.")
                return

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logging.info(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f}")

    # 검증
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch).cpu().numpy().flatten()
            output = torch.sigmoid(torch.tensor(output)).numpy()
            preds.extend(output)
            trues.extend(y_batch.numpy())

    logging.info(f"✅ Validation LogLoss: {log_loss(trues, preds):.4f}")
    logging.info(f"✅ Validation ROC AUC: {roc_auc_score(trues, preds):.4f}")

    # 테스트셋 처리
    test_df = pd.read_csv(TEST_PATH)
    X_test_raw = test_df.drop(columns=["index"])
    X_test_raw = clip_numeric_features(X_test_raw)
    X_test = pd.get_dummies(X_test_raw)

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
            output = torch.sigmoid(torch.tensor(output)).numpy()
            predictions.extend(output)

    predictions = np.array(predictions)
    if np.isnan(predictions).any():
        logging.warning("NaN found in predictions. Replacing with 0.5.")
    predictions = np.nan_to_num(predictions, nan=0.5)

    os.makedirs("submissions", exist_ok=True)
    submit_df = pd.DataFrame({
        "index": test_df["index"].values,
        "k": predictions
    })
    submit_df.to_csv(SUBMIT_PATH, index=False)
    logging.info(f"📁 Submission saved to {SUBMIT_PATH}")

if __name__ == "__main__":
    main()
