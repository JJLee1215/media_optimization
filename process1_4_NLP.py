"""
7_mlp_model.py
CPU에서 돌아가는 얇은 MLP로 titer 예측

구조: Input → 32 → 16 → 1
파라미터 수: ~865개 (매우 작음)
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

DATA_PATH = "data_file/batch_table.csv"
OUT_DIR   = "outputs/mlp"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

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
X_COLS = [c.split("(")[0].strip() for c in INPUT_COLS]


# ── 모델 ──────────────────────────────────────────────────────────────────────
class ThinMLP(nn.Module):
    """
    Input → 32 → 16 → 1
    BatchNorm + Dropout으로 작은 데이터셋 과적합 방지
    """
    def __init__(self, input_dim, hidden_dims=None, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── 데이터 ────────────────────────────────────────────────────────────────────
def load(path):
    df = pd.read_csv(path)
    x_cols = [c for c in X_COLS if c in df.columns]
    X = df[x_cols].values.astype(np.float32)
    y = df["titer_final"].values.astype(np.float32)
    print(f"  X: {X.shape}   Y: {y.shape}")
    return X, y, x_cols


def prepare(X, y, test_size=0.2, val_size=0.1):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=42
    )
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio, random_state=42
    )
    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val   = x_scaler.transform(X_val)
    X_test  = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s   = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

    def to_loader(Xd, yd, shuffle=False):
        ds = TensorDataset(
            torch.tensor(Xd.astype(np.float32)),
            torch.tensor(yd.astype(np.float32))
        )
        return DataLoader(ds, batch_size=16, shuffle=shuffle)

    loaders = {
        "train": to_loader(X_train, y_train_s, shuffle=True),
        "val":   to_loader(X_val,   y_val_s),
    }
    return loaders, x_scaler, y_scaler, X_test, y_test


# ── 학습 ──────────────────────────────────────────────────────────────────────
def train(model, loaders, epochs=200, lr=1e-3, out_dir=OUT_DIR):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5
    )
    criterion = nn.MSELoss()
    history   = {"train": [], "val": []}
    best_val, best_epoch = float("inf"), 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for xb, yb in loaders["train"]:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(xb)
        tl /= len(loaders["train"].dataset)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in loaders["val"]:
                vl += criterion(model(xb), yb).item() * len(xb)
        vl /= len(loaders["val"].dataset)

        scheduler.step(vl)
        history["train"].append(tl)
        history["val"].append(vl)

        if vl < best_val:
            best_val, best_epoch = vl, epoch
            torch.save(model.state_dict(), f"{out_dir}/best_model.pt")

        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3} | train {tl:.4f} | val {vl:.4f} | best@{best_epoch}")

    print(f"\n[학습 완료]  {time.time()-t0:.1f}초  best val loss: {best_val:.4f} (epoch {best_epoch})")
    return history


# ── 평가 ──────────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, y_scaler, out_dir):
    model.eval()
    with torch.no_grad():
        y_pred_s = model(torch.tensor(X_test.astype(np.float32))).numpy()
    y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n[테스트 결과]")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_test, y_pred, alpha=0.6, s=30)
    lims = [min(y_test.min(), y_pred.min()) - 2,
            max(y_test.max(), y_pred.max()) + 2]
    axes[0].plot(lims, lims, "r--", lw=1)
    axes[0].set_xlabel("Actual titer")
    axes[0].set_ylabel("Predicted titer")
    axes[0].set_title(f"MLP  (R²={r2:.3f})")

    axes[1].hist(y_pred - y_test, bins=15, edgecolor="white", color="#CECBF6")
    axes[1].axvline(0, color="red", lw=1, linestyle="--")
    axes[1].set_title("Residuals")
    axes[1].set_xlabel("Predicted - Actual")

    plt.tight_layout()
    out = f"{out_dir}/eval.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[그래프] {out}")
    return rmse, r2


def plot_history(history, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train"], label="train")
    ax.plot(history["val"],   label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (scaled)")
    ax.set_title("Training curve")
    ax.legend()
    plt.tight_layout()
    out = f"{out_dir}/training_curve.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[그래프] {out}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[로드] {DATA_PATH}")
    X, y, x_cols = load(DATA_PATH)

    loaders, x_scaler, y_scaler, X_test, y_test = prepare(X, y)

    model = ThinMLP(input_dim=len(x_cols), hidden_dims=[32, 16], dropout=0.1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[모델]  Input({len(x_cols)}) → 32 → 16 → 1")
    print(f"  파라미터 수: {n_params:,}")

    print("\n[학습 시작]")
    history = train(model, loaders, epochs=200, lr=1e-3)
    plot_history(history, OUT_DIR)

    model.load_state_dict(torch.load(f"{OUT_DIR}/best_model.pt", weights_only=True))
    rmse, r2 = evaluate(model, X_test, y_test, y_scaler, OUT_DIR)

    result = {"rmse": round(float(rmse), 4), "r2": round(float(r2), 4),
              "n_params": n_params}
    with open(f"{OUT_DIR}/result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[완료] outputs/mlp/ 에서 결과 확인")