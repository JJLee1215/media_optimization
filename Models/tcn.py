"""
Models/tcn.py
TCN (Temporal Convolutional Network) Regression

Input:
  X_train : (n_train, T, d_dynamic)   time series (scaled)
  X_test  : (n_test,  T, d_dynamic)

  T         = 14 (days)
  d_dynamic = 28
    d_dyn_media(9)   + d_dyn_feed(4) + d_dyn_process(15)

Target:
  y_train : (n_train,)   titer_final
  y_test  : (n_test,)

Source: timeseries_syn.csv  via data_preprocess.get_timeseries_data()

Network:
  Input(batch, T, 28)
    → TemporalBlock × N (Dilated Causal Conv1d + Residual)
      dilation: 1, 2, 4, ...  (2^i per layer)
      kernel_size: TCN_KERNEL_SIZE (default 3)
      channels: TCN_NUM_CHANNELS (default [32, 32, 32])
    → 마지막 타임스텝 [:, -1, :]
    → Linear(channel → 1)
    → Output(batch,)

TCN의 장점:
  - LSTM/RNN과 달리 병렬 처리 → 학습 빠름
  - Dilated conv로 장기 의존성 안정적으로 처리
    (receptive field = (kernel_size-1) * sum(dilations))
  - Residual connection으로 깊어도 학습 안정적
  - 데이터 적을 때 LSTM보다 상대적으로 유리

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

MODEL_NAME = "tcn"
SAVE_PATH  = config.model_save_path(MODEL_NAME)
RESULT_DIR = config.result_dir(MODEL_NAME)


# ══════════════════════════════════════════════
# Network components
# ══════════════════════════════════════════════

class _CausalConv1d(nn.Module):
    """
    Causal (left-padded) Conv1d.
    미래 정보를 보지 않도록 왼쪽에만 패딩을 추가한 1D 컨볼루션.
    dilated convolution과 조합해 긴 시계열의 의존성을 효율적으로 포착.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation   # causal padding
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x):
        # x: (batch, channels, T)
        # 왼쪽에만 패딩이 들어갔으므로 오른쪽 self.padding만큼 잘라냄 → causal 보장
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding else out


class _TemporalBlock(nn.Module):
    """
    TCN의 기본 빌딩 블록.

    구조:
      CausalConv1d → LayerNorm → ReLU → Dropout
      CausalConv1d → LayerNorm → ReLU → Dropout
      + Residual connection (채널 수 다르면 1×1 conv로 맞춤)

    residual connection 덕분에 레이어가 깊어져도 그래디언트 소실 없이 학습 가능.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = _CausalConv1d(in_channels,  out_channels, kernel_size, dilation)
        self.conv2 = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)

        self.norm1   = nn.LayerNorm(out_channels)
        self.norm2   = nn.LayerNorm(out_channels)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 채널 수가 다를 경우 residual을 맞추기 위한 1×1 conv
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else None
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.conv.weight)
        nn.init.kaiming_normal_(self.conv2.conv.weight)

    def forward(self, x):
        # x: (batch, in_channels, T)
        res = x if self.residual_conv is None else self.residual_conv(x)

        # 첫 번째 conv 블록
        out = self.conv1(x)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.dropout(out)

        # 두 번째 conv 블록
        out = self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.dropout(out)

        return self.relu(out + res)   # residual connection


class _TCNNet(nn.Module):
    """
    TCN 전체 네트워크.

    num_channels : 레이어별 채널 수 리스트 (예: [32, 32, 32])
    dilation은 레이어마다 2^i로 지수 증가 → receptive field 확장

    마지막 타임스텝의 출력으로 titer 예측 (many-to-one).
    """
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super().__init__()
        layers = []
        in_ch  = input_size

        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i   # 1, 2, 4, 8, ...
            layers.append(
                _TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.fc      = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x: (batch, T, input_size)
        x = x.transpose(1, 2)             # → (batch, input_size, T)
        x = self.network(x)               # → (batch, last_channel, T)
        x = x[:, :, -1]                   # 마지막 타임스텝 → (batch, last_channel)
        return self.fc(x).squeeze(-1)     # → (batch,)


# ══════════════════════════════════════════════
# Model class
# ══════════════════════════════════════════════

class TCNModel:
    def __init__(self):
        self.net      = None
        self.y_scaler = StandardScaler()
        self.device   = config.DEVICE
        self.history  = {"train": [], "val": []}

    def train(self, loaders, y_scaler):
        """
        loaders  : {"train": DataLoader, "val": DataLoader}
                   DataLoader yields (X_batch, y_batch)
                   X_batch: (batch, T, d_dynamic)   already scaled
                   y_batch: (batch,)                already y_scaler transformed
        y_scaler : fitted StandardScaler for y (for inverse transform at predict)
        """
        self.y_scaler = y_scaler

        # input_size는 DataLoader에서 자동 결정
        sample_x, _ = next(iter(loaders["train"]))
        input_size   = sample_x.shape[2]   # d_dynamic = 28

        self.net = _TCNNet(
            input_size   = input_size,
            num_channels = config.TCN_NUM_CHANNELS,
            kernel_size  = config.TCN_KERNEL_SIZE,
            dropout      = config.TCN_DROPOUT,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=config.TCN_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()

        best_val, best_epoch = float("inf"), 0

        n_params = sum(p.numel() for p in self.net.parameters())
        print(f"[TCN] Training...  input_size={input_size}  params={n_params:,}")
        print(f"[TCN] Channels={config.TCN_NUM_CHANNELS}  kernel={config.TCN_KERNEL_SIZE}  dilation=1,2,4,...")
        t0 = time.time()

        for epoch in range(1, config.TCN_EPOCHS + 1):
            # ── train ──
            self.net.train()
            train_loss = 0.0
            for xb, yb in loaders["train"]:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(loaders["train"].dataset)

            # ── val ──
            self.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in loaders["val"]:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred    = self.net(xb)
                    val_loss += criterion(pred, yb).item() * len(xb)
            val_loss /= len(loaders["val"].dataset)

            scheduler.step(val_loss)
            self.history["train"].append(train_loss)
            self.history["val"].append(val_loss)

            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch
                torch.save(self.net.state_dict(), SAVE_PATH.parent / "tcn_tmp.pt")

            if epoch % 20 == 0 or epoch == 1:
                print(f"  epoch {epoch:>3} | train {train_loss:.4f} | val {val_loss:.4f} | best@{best_epoch}")

        # best 가중치 복원
        self.net.load_state_dict(
            torch.load(SAVE_PATH.parent / "tcn_tmp.pt", weights_only=True)
        )
        print(f"[TCN] Training complete.  {time.time()-t0:.1f}s  best val={best_val:.4f}@{best_epoch}")

        # ── 학습 정보 기록용: 학습 후에만 확정되는 값 저장 ──
        self.best_epoch = best_epoch
        self.best_val   = round(float(best_val), 4)

    def predict(self, X):
        """
        X       : (n, T, d_dynamic)  numpy array (scaled)
        Returns : (n,)               original scale
        """
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
        print(f"[TCN] RMSE: {rmse:.4f}  R²: {r2:.4f}")

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        # Predicted vs Actual
        axes[0].scatter(y_test, y_pred, alpha=0.6, s=30, color="#1D9E75")
        lims = [min(y_test.min(), y_pred.min()) - 0.2,
                max(y_test.max(), y_pred.max()) + 0.2]
        axes[0].plot(lims, lims, "r--", lw=1)
        axes[0].set_xlabel("Actual")
        axes[0].set_ylabel("Predicted")
        axes[0].set_title(f"TCN  (R²={r2:.3f}  RMSE={rmse:.3f})")
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
        print(f"[TCN] Plot saved: {RESULT_DIR}/eval.png")

        return {
            "model": MODEL_NAME,
            "rmse" : round(float(rmse), 4),
            "r2"   : round(float(r2),   4),
        }

    def get_config(self):
        """
        학습 정보 기록용 하이퍼파라미터 리포트.
        train.py의 train_model()이 result.json의 meta.hyperparams에
        이 값을 그대로 저장함.

        ※ num_channels/kernel_size/dropout/lr/epoch는 config.py에서
          미리 고정된 값. best_epoch/best_val은 train()이 끝난 뒤에만
          알 수 있는 값이라 getattr로 방어적으로 조회함.
        """
        return {
            "epoch"        : config.TCN_EPOCHS,
            "num_channels" : config.TCN_NUM_CHANNELS,
            "kernel_size"  : config.TCN_KERNEL_SIZE,
            "dropout"      : config.TCN_DROPOUT,
            "lr"           : config.TCN_LR,
            "best_epoch"   : getattr(self, "best_epoch", None),
            "best_val"     : getattr(self, "best_val", None),
        }

    def save(self):
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict"  : self.net.state_dict(),
            "y_scaler"    : self.y_scaler,
            "num_channels": config.TCN_NUM_CHANNELS,
            "kernel_size" : config.TCN_KERNEL_SIZE,
            "dropout"     : config.TCN_DROPOUT,
        }, SAVE_PATH)
        print(f"[TCN] Saved: {SAVE_PATH}")

    def load(self):
        ckpt = torch.load(SAVE_PATH, map_location=self.device, weights_only=False)
        # 저장된 하이퍼파라미터로 네트워크 재구성
        # input_size는 저장 안 했으므로 config 기본값(28)으로 복원
        # 필요 시 save()에 input_size도 저장하도록 확장 가능
        self.net = _TCNNet(
            input_size   = 28,
            num_channels = ckpt["num_channels"],
            kernel_size  = ckpt["kernel_size"],
            dropout      = ckpt["dropout"],
        ).to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        self.y_scaler = ckpt["y_scaler"]
        print(f"[TCN] Loaded: {SAVE_PATH}")


if __name__ == "__main__":
    from data_preprocess import get_timeseries_data

    loaders, x_sc, y_sc, X_test, y_test, feat_cols = get_timeseries_data()
    model = TCNModel()
    model.train(loaders, y_sc)
    model.evaluate(X_test, y_test)
    model.save()