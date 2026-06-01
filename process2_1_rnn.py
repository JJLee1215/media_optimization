"""
ts1_rnn.py
Vanilla RNN으로 시계열 → 최종 titer 예측

X: (배치수, 시점수, 10변수) 시계열
Y: 각 배치의 최종 titer (g/L)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json, time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "data_file/IndPenSim_Optimized_Final.csv"
OUT_DIR   = "outputs/time_series/rnn"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

BATCH_COL = "Batch ID"
FAULT_COL = "Fault flag"
TIME_COL  = "Time (h)"
TARGET    = "Penicillin concentration(P:g/L)"

INPUT_COLS = [
    "Aeration rate(Fg:L/h)",
    "Agitator RPM(RPM:RPM)",
    "Sugar feed rate(Fs:L/h)",
    "Acid flow rate(Fa:L/h)",
    "Base flow rate(Fb:L/h)",
    "Heating/cooling water flow rate(Fc:L/h)",
    "Heating water flow rate(Fh:L/h)",
    "Water for injection/dilution(Fw:L/h)",
    "PAA flow(Fpaa:PAA flow (L/h))",
    "Oil flow(Foil:L/hr)",
]


class ThinRNN(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def load(path):
    df = pd.read_csv(path)
    raman = [c for c in df.columns if c.startswith("R_Bin_")]
    df = df.drop(columns=raman)
    if FAULT_COL in df.columns:
        df = df[df[FAULT_COL] == 0]
    return df


def make_sequences(df):
    exist_cols = [c for c in INPUT_COLS if c in df.columns]
    X_list, y_list = [], []
    for _, grp in df.groupby(BATCH_COL):
        grp = grp.sort_values(TIME_COL)
        X_list.append(grp[exist_cols].values)
        y_list.append(grp[TARGET].iloc[-1])
    seq_len = 100
    X = np.array([x[:seq_len] for x in X_list], dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"  X: {X.shape}  (배치수, 시점수, 변수수)")
    print(f"  Y: {y.shape}")
    return X, y, exist_cols


def prepare(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    n_feat = X.shape[2]
    x_scaler = StandardScaler()
    x_scaler.fit(X_train.reshape(-1, n_feat))
    X_train = x_scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_val   = x_scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
    X_test  = x_scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1,1)).ravel()
    y_val_s   = y_scaler.transform(y_val.reshape(-1,1)).ravel()

    def to_loader(Xd, yd, shuffle=False):
        return DataLoader(TensorDataset(torch.tensor(Xd), torch.tensor(yd)),
                          batch_size=16, shuffle=shuffle)
    loaders = {
        "train": to_loader(X_train, y_train_s, shuffle=True),
        "val":   to_loader(X_val, y_val_s),
    }
    return loaders, x_scaler, y_scaler, X_test, y_test


def train(model, loaders, epochs=800, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    best_val, best_epoch = float("inf"), 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        tl = sum(criterion(model(xb), yb).item() * len(xb)
                 for xb, yb in loaders["train"]) / len(loaders["train"].dataset)
        model.eval()
        with torch.no_grad():
            vl = sum(criterion(model(xb), yb).item() * len(xb)
                     for xb, yb in loaders["val"]) / len(loaders["val"].dataset)
        scheduler.step(vl)
        if vl < best_val:
            best_val, best_epoch = vl, epoch
            torch.save(model.state_dict(), f"{OUT_DIR}/best_model.pt")
        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3} | train {tl:.4f} | val {vl:.4f} | best@{best_epoch}")

    print(f"\n[완료] {time.time()-t0:.1f}초  best val: {best_val:.4f}")


def evaluate(model, X_test, y_test, y_scaler):
    model.eval()
    with torch.no_grad():
        y_pred_s = model(torch.tensor(X_test)).numpy()
    y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1,1)).ravel()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"  RMSE: {rmse:.4f}  R²: {r2:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_test, y_pred, alpha=0.6, s=30)
    lims = [min(y_test.min(), y_pred.min())-2, max(y_test.max(), y_pred.max())+2]
    axes[0].plot(lims, lims, "r--", lw=1)
    axes[0].set_title(f"RNN  (R²={r2:.3f})")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[1].hist(y_pred - y_test, bins=15, edgecolor="white", color="#F0997B")
    axes[1].axvline(0, color="red", lw=1, ls="--"); axes[1].set_title("Residuals")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/eval.png", dpi=120); plt.close()
    return rmse, r2


if __name__ == "__main__":
    df = load(DATA_PATH)
    X, y, feat_cols = make_sequences(df)
    loaders, x_scaler, y_scaler, X_test, y_test = prepare(X, y)

    model = ThinRNN(input_size=len(feat_cols), hidden_size=32)
    print(f"[모델] RNN  파라미터: {sum(p.numel() for p in model.parameters()):,}")

    train(model, loaders, epochs=800)
    model.load_state_dict(torch.load(f"{OUT_DIR}/best_model.pt", weights_only=True))
    rmse, r2 = evaluate(model, X_test, y_test, y_scaler)

    with open(f"{OUT_DIR}/result.json", "w") as f:
        json.dump({"model":"RNN","rmse":round(float(rmse),4),"r2":round(float(r2),4)}, f, indent=2)
    print(f"[완료] {OUT_DIR}/")