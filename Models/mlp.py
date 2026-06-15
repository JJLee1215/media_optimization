"""
Models/Model_MLP.py
MLP (Multi-Layer Perceptron) — 정적 입력
입력: (n_samples, n_features) 2D
"""

import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config

cfg = Config()
MODEL_NAME = "mlp"
SAVE_PATH  = cfg.model_save_path(MODEL_NAME)
RESULT_DIR = cfg.result_dir(MODEL_NAME)


class _ThinMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPModel:
    def __init__(self):
        self.net      = None
        self.y_scaler = StandardScaler()
        self.device   = cfg.DEVICE

    def _build(self, input_dim):
        return _ThinMLP(input_dim, cfg.MLP_HIDDEN_DIMS, cfg.MLP_DROPOUT).to(self.device)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        n = len(X_train)
        if X_val is None:
            split = int(n * 0.9)
            X_val, y_val = X_train[split:], y_train[split:]
            X_train, y_train = X_train[:split], y_train[:split]

        y_train_s = self.y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_s   = self.y_scaler.transform(y_val.reshape(-1, 1)).ravel()

        def to_loader(X, y, shuffle=False):
            ds = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32)
            )
            return DataLoader(ds, batch_size=cfg.MLP_BATCH_SIZE, shuffle=shuffle)

        loaders = {
            "train": to_loader(X_train, y_train_s, shuffle=True),
            "val":   to_loader(X_val, y_val_s),
        }

        self.net = self._build(X_train.shape[1])
        optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.MLP_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        criterion = nn.MSELoss()
        best_val, best_epoch = float("inf"), 0

        print(f"[MLP] 학습 시작  train={len(X_train)}  val={len(X_val)}")
        t0 = time.time()
        self.history = {"train": [], "val": []}

        for epoch in range(1, cfg.MLP_EPOCHS + 1):
            self.net.train()
            tl = sum(criterion(self.net(xb.to(self.device)), yb.to(self.device)).item() * len(xb)
                     for xb, yb in loaders["train"]) / len(loaders["train"].dataset)
            self.net.eval()
            with torch.no_grad():
                vl = sum(criterion(self.net(xb.to(self.device)), yb.to(self.device)).item() * len(xb)
                         for xb, yb in loaders["val"]) / len(loaders["val"].dataset)
            scheduler.step(vl)
            self.history["train"].append(tl)
            self.history["val"].append(vl)

            if vl < best_val:
                best_val, best_epoch = vl, epoch
                torch.save(self.net.state_dict(), SAVE_PATH.parent / "mlp_tmp.pt")

            if epoch % 50 == 0 or epoch == 1:
                print(f"  epoch {epoch:>3} | train {tl:.4f} | val {vl:.4f} | best@{best_epoch}")

        self.net.load_state_dict(torch.load(SAVE_PATH.parent / "mlp_tmp.pt", weights_only=True))
        print(f"[MLP] 완료  {time.time()-t0:.1f}초  best val: {best_val:.4f}@{best_epoch}")

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            y_s = self.net(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy()
        return self.y_scaler.inverse_transform(y_s.reshape(-1, 1)).ravel(), None

    def evaluate(self, X_test, y_test):
        y_pred, _ = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        print(f"[MLP] RMSE: {rmse:.4f}  R²: {r2:.4f}")

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(y_test, y_pred, alpha=0.6, s=30)
        lims = [min(y_test.min(), y_pred.min())-1, max(y_test.max(), y_pred.max())+1]
        axes[0].plot(lims, lims, "r--", lw=1)
        axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
        axes[0].set_title(f"MLP  (R²={r2:.3f}  RMSE={rmse:.3f})")
        axes[1].plot(self.history["train"], label="train")
        axes[1].plot(self.history["val"],   label="val", ls="--")
        axes[1].set_title("Training curve"); axes[1].legend()
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "eval.png", dpi=120)
        plt.close()
        return {"model": MODEL_NAME, "rmse": round(float(rmse), 4), "r2": round(float(r2), 4)}

    def save(self):
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "y_scaler":   self.y_scaler,
            "input_dim":  next(self.net.parameters()).shape[1],
        }, SAVE_PATH)
        print(f"[MLP] 저장: {SAVE_PATH}")

    def load(self):
        ckpt = torch.load(SAVE_PATH, map_location=self.device, weights_only=False)
        self.net = self._build(ckpt["input_dim"])
        self.net.load_state_dict(ckpt["state_dict"])
        self.y_scaler = ckpt["y_scaler"]
        print(f"[MLP] 로드: {SAVE_PATH}")


if __name__ == "__main__":
    from data import get_static_data
    X_train, X_test, y_train, y_test, x_cols, scaler = get_static_data(use_syn=True)
    model = MLPModel()
    model.train(X_train, y_train)
    model.evaluate(X_test, y_test)
    model.save()