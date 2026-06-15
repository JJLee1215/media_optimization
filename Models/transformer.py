"""
Models/Model_Transformer.py
Transformer Encoder — 시계열 입력
"""

import numpy as np
import time, math
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config

cfg = Config()
MODEL_NAME = "transformer"
SAVE_PATH  = cfg.model_save_path(MODEL_NAME)
RESULT_DIR = cfg.result_dir(MODEL_NAME)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class _ThinTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.proj    = nn.Linear(input_size, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*2, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc      = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.fc(x.mean(dim=1)).squeeze(-1)


class TransformerModel:
    def __init__(self):
        self.net      = None
        self.y_scaler = None
        self.device   = cfg.DEVICE
        self.history  = {"train": [], "val": []}

    def train(self, loaders, y_scaler):
        self.y_scaler = y_scaler
        input_size = next(iter(loaders["train"]))[0].shape[2]
        self.net   = _ThinTransformer(
            input_size, cfg.TF_D_MODEL, cfg.TF_NHEAD,
            cfg.TF_NUM_LAYERS, cfg.TF_DROPOUT
        ).to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.TF_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()
        best_val, best_epoch = float("inf"), 0

        print(f"[Transformer] 학습 시작  파라미터: {sum(p.numel() for p in self.net.parameters()):,}")
        t0 = time.time()

        for epoch in range(1, cfg.TF_EPOCHS + 1):
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
                torch.save(self.net.state_dict(), SAVE_PATH.parent / "transformer_tmp.pt")

            if epoch % 20 == 0 or epoch == 1:
                print(f"  epoch {epoch:>3} | train {tl:.4f} | val {vl:.4f} | best@{best_epoch}")

        self.net.load_state_dict(torch.load(SAVE_PATH.parent / "transformer_tmp.pt", weights_only=True))
        print(f"[Transformer] 완료  {time.time()-t0:.1f}초  best val: {best_val:.4f}@{best_epoch}")

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            y_s = self.net(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy()
        return self.y_scaler.inverse_transform(y_s.reshape(-1, 1)).ravel(), None

    def evaluate(self, X_test, y_test):
        y_pred, _ = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        print(f"[Transformer] RMSE: {rmse:.4f}  R²: {r2:.4f}")

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(y_test, y_pred, alpha=0.6, s=30)
        lims = [min(y_test.min(), y_pred.min())-1, max(y_test.max(), y_pred.max())+1]
        axes[0].plot(lims, lims, "r--", lw=1)
        axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
        axes[0].set_title(f"Transformer  (R²={r2:.3f}  RMSE={rmse:.3f})")
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
                    "input_size": self.net.proj.in_features}, SAVE_PATH)
        print(f"[Transformer] 저장: {SAVE_PATH}")

    def load(self):
        ckpt = torch.load(SAVE_PATH, map_location=self.device, weights_only=False)
        self.net = _ThinTransformer(
            ckpt["input_size"], cfg.TF_D_MODEL, cfg.TF_NHEAD,
            cfg.TF_NUM_LAYERS, cfg.TF_DROPOUT
        ).to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        self.y_scaler = ckpt["y_scaler"]
        print(f"[Transformer] 로드: {SAVE_PATH}")


if __name__ == "__main__":
    from data import get_timeseries_data
    loaders, x_sc, y_sc, X_test, y_test = get_timeseries_data(use_syn=True)
    model = TransformerModel()
    model.train(loaders, y_sc)
    model.evaluate(X_test, y_test)
    model.save()