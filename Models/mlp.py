"""
Models/mlp.py
MLP (Multi-Layer Perceptron) Regression

Input:
  X_train : (n_train, 9)   initial media composition (scaled)
  X_test  : (n_test,  9)

  Features (9):
    Glucose_0, Glutamine_0, Asparagine_0
    Lactate_0, Ammonia_0
    Cu_0, Zn_0, Mn_0, Fe_0

Target:
  y_train : (n_train,)   titer_final
  y_test  : (n_test,)

Source: batch_table_syn.csv  via data_preprocess.get_static_data()

Network:
  Input(9) → Linear(32) → BN → ReLU → Dropout
           → Linear(16) → BN → ReLU → Dropout
           → Linear(1)

Methods:
  train()        fit model (epoch loop)
  predict()      return predictions
  evaluate()     compute RMSE, R² + save plots
  save() / load() persist model to disk
"""

import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

MODEL_NAME = "mlp"
SAVE_PATH  = config.model_save_path(MODEL_NAME)
RESULT_DIR = config.result_dir(MODEL_NAME)


# ── Network ───────────────────────────────────

class _MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
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


# ── Model ────────────────────────────────────

class MLPModel:
    def __init__(self):
        self.net      = None
        self.y_scaler = StandardScaler()
        self.scaler   = None
        self.x_cols   = None
        self.device   = config.DEVICE
        self.history  = {"train": [], "val": []}

    def train(self, X_train, y_train, x_cols=None, scaler=None):
        """
        X_train : (n_train, 9)  already scaled
        y_train : (n_train,)
        x_cols  : feature names
        scaler  : fitted StandardScaler (saved for inference)
        """
        self.scaler = scaler
        self.x_cols = x_cols

        # val split from train
        split    = int(len(X_train) * 0.9)
        X_val    = X_train[split:]
        y_val    = y_train[split:]
        X_tr     = X_train[:split]
        y_tr     = y_train[:split]

        # scale y
        y_tr_s = self.y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
        y_val_s = self.y_scaler.transform(y_val.reshape(-1, 1)).ravel()

        def to_loader(X, y, shuffle=False):
            ds = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
            )
            return DataLoader(ds, batch_size=config.MLP_BATCH_SIZE, shuffle=shuffle)

        loaders = {
            "train": to_loader(X_tr,  y_tr_s,  shuffle=True),
            "val"  : to_loader(X_val, y_val_s),
        }

        self.net = _MLPNet(X_train.shape[1], config.MLP_HIDDEN_DIMS, config.MLP_DROPOUT).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=config.MLP_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        criterion = nn.MSELoss()
        best_val, best_epoch = float("inf"), 0

        print(f"[MLP] Training...  n_train={len(X_tr)}  n_val={len(X_val)}")
        print(f"[MLP] Params: {sum(p.numel() for p in self.net.parameters()):,}")
        t0 = time.time()

        for epoch in range(1, config.MLP_EPOCHS + 1):
            self.net.train()
            tl = sum(
                criterion(self.net(xb.to(self.device)), yb.to(self.device)).item() * len(xb)
                for xb, yb in loaders["train"]
            ) / len(loaders["train"].dataset)

            self.net.eval()
            with torch.no_grad():
                vl = sum(
                    criterion(self.net(xb.to(self.device)), yb.to(self.device)).item() * len(xb)
                    for xb, yb in loaders["val"]
                ) / len(loaders["val"].dataset)

            scheduler.step(vl)
            self.history["train"].append(tl)
            self.history["val"].append(vl)

            if vl < best_val:
                best_val, best_epoch = vl, epoch
                torch.save(self.net.state_dict(), SAVE_PATH.parent / "mlp_tmp.pt")

            if epoch % 50 == 0 or epoch == 1:
                print(f"  epoch {epoch:>3} | train {tl:.4f} | val {vl:.4f} | best@{best_epoch}")

        self.net.load_state_dict(
            torch.load(SAVE_PATH.parent / "mlp_tmp.pt", weights_only=True)
        )
        print(f"[MLP] Training complete.  {time.time()-t0:.1f}s  best val: {best_val:.4f}@{best_epoch}")

    def predict(self, X):
        """Returns (y_pred, None)."""
        self.net.eval()
        with torch.no_grad():
            y_s = self.net(
                torch.tensor(X, dtype=torch.float32).to(self.device)
            ).cpu().numpy()
        return self.y_scaler.inverse_transform(y_s.reshape(-1, 1)).ravel(), None

    def evaluate(self, X_test, y_test):
        """Compute RMSE, R² and save evaluation plots."""
        y_pred, _ = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        print(f"[MLP] RMSE: {rmse:.4f}  R²: {r2:.4f}")

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        # Predicted vs Actual
        axes[0].scatter(y_test, y_pred, alpha=0.6, s=30, color="#1D9E75")
        lims = [min(y_test.min(), y_pred.min()) - 0.2,
                max(y_test.max(), y_pred.max()) + 0.2]
        axes[0].plot(lims, lims, "r--", lw=1)
        axes[0].set_xlabel("Actual")
        axes[0].set_ylabel("Predicted")
        axes[0].set_title(f"MLP  (R²={r2:.3f}  RMSE={rmse:.3f})")
        axes[0].grid(True, alpha=0.3)

        # Training curve
        axes[1].plot(self.history["train"], label="train", color="#1D9E75")
        axes[1].plot(self.history["val"],   label="val",   color="#534AB7", ls="--")
        axes[1].set_title("Training curve")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MSE Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(RESULT_DIR / "eval.png", dpi=120)
        plt.close()
        print(f"[MLP] Plot saved: {RESULT_DIR}/eval.png")

        return {
            "model": MODEL_NAME,
            "rmse" : round(float(rmse), 4),
            "r2"   : round(float(r2),   4),
        }

    def save(self):
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "y_scaler"  : self.y_scaler,
            "scaler"    : self.scaler,
            "x_cols"    : self.x_cols,
            "input_dim" : next(self.net.parameters()).shape[1],
        }, SAVE_PATH)
        print(f"[MLP] Saved: {SAVE_PATH}")

    def load(self):
        ckpt = torch.load(SAVE_PATH, map_location=self.device, weights_only=False)
        self.net = _MLPNet(
            ckpt["input_dim"], config.MLP_HIDDEN_DIMS, config.MLP_DROPOUT
        ).to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        self.y_scaler = ckpt["y_scaler"]
        self.scaler   = ckpt["scaler"]
        self.x_cols   = ckpt["x_cols"]
        print(f"[MLP] Loaded: {SAVE_PATH}")


if __name__ == "__main__":
    from data_preprocess import get_static_data
    X_train, X_test, y_train, y_test, x_cols, scaler = get_static_data()

    model = MLPModel()
    model.train(X_train, y_train, x_cols=x_cols, scaler=scaler)
    model.evaluate(X_test, y_test)
    model.save()