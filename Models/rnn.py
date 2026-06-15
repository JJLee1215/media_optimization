"""
Models/Model_RNN.py
Vanilla RNN — 시계열 입력
입력: (n_samples, T, n_features) 3D
"""

import numpy as np
import time
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config

cfg = Config()
MODEL_NAME = "rnn"
SAVE_PATH  = cfg.model_save_path(MODEL_NAME)
RESULT_DIR = cfg.result_dir(MODEL_NAME)


class _ThinRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class RNNModel:
    def __init__(self):
        self.net      = None
        self.y_scaler = None
        self.device   = cfg.DEVICE
        self.history  = {"train": [], "val": []}

    def train(self, loaders, y_scaler):
        self.y_scaler = y_scaler
        input_size = next(iter(loaders["train"]))[0].shape[2]
        self.net   = _ThinRNN(input_size, cfg.RNN_HIDDEN_SIZE, cfg.RNN_NUM_LAYERS).to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.RNN_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()
        best_val, best_epoch = float("inf"), 0

        print(f"[RNN] 학습 시작  파라미터: {sum(p.numel() for p in self.net.parameters()):,}")
        t0 = time.time()

        for epoch in range(1, cfg.RNN_EPOCHS + 1):
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
                torch.save(self.net.state_dict(), SAVE_PATH.parent / "rnn_tmp.pt")

            if epoch % 20 == 0 or epoch == 1:
                print(f"  epoch {epoch:>3} | train {tl:.4f} | val {vl:.4f} | best@{best_epoch}")

        self.net.load_state_dict(torch.load(SAVE_PATH.parent / "rnn_tmp.pt", weights_only=True))
        print(f"[RNN] 완료  {time.time()-t0:.1f}초  best val: {best_val:.4f}@{best_epoch}")

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            y_s = self.net(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy()
        return self.y_scaler.inverse_transform(y_s.reshape(-1, 1)).ravel(), None

    def evaluate(self, X_test, y_test):
        y_pred, _ = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        print(f"[RNN] RMSE: {rmse:.4f}  R²: {r2:.4f}")

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(y_test, y_pred, alpha=0.6, s=30)
        lims = [min(y_test.min(), y_pred.min())-1, max(y_test.max(), y_pred.max())+1]
        axes[0].plot(lims, lims, "r--", lw=1)
        axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
        axes[0].set_title(f"RNN  (R²={r2:.3f}  RMSE={rmse:.3f})")
        axes[1].plot(self.history["train"], label="train")
        axes[1].plot(self.history["val"],   label="val", ls="--")
        axes[1].set_title("Training curve"); axes[1].legend()
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "eval.png", dpi=120)
        plt.close()
        return {"model": MODEL_NAME, "rmse": round(float(rmse), 4), "r2": round(float(r2), 4)}

    def save(self):
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "y_scaler": self.y_scaler,
                    "input_size": self.net.rnn.input_size}, SAVE_PATH)
        print(f"[RNN] 저장: {SAVE_PATH}")

    def load(self):
        ckpt = torch.load(SAVE_PATH, map_location=self.device, weights_only=False)
        self.net = _ThinRNN(ckpt["input_size"], cfg.RNN_HIDDEN_SIZE, cfg.RNN_NUM_LAYERS).to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        self.y_scaler = ckpt["y_scaler"]
        print(f"[RNN] 로드: {SAVE_PATH}")


if __name__ == "__main__":
    from data import get_timeseries_data
    loaders, x_sc, y_sc, X_test, y_test = get_timeseries_data(use_syn=True)
    model = RNNModel()
    model.train(loaders, y_sc)
    model.evaluate(X_test, y_test)
    model.save()