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

  ※ Heterogeneity pipeline (use_pipeline=True) 적용 시:
     X_train/X_test는 (n, 230)으로 변환된 상태로 들어옴
     (SMILES → RDKit descriptor → GEM 벡터 → Mean Pooling)
     Input(9) → ... 구조가 자동으로 Input(230) → ... 으로 바뀜
     (_MLPNet이 X_train.shape[1]을 받아 동적으로 구성되기 때문)

Target:
  y_train : (n_train,)   titer_final
  y_test  : (n_test,)

Source: batch_table_syn.csv  via data_preprocess.get_static_data()

Network:
  Input(9 or 230) → Linear(32) → BN → ReLU → Dropout
                   → Linear(16) → BN → ReLU → Dropout
                   → Linear(1)

Methods:
  train()        fit model (epoch loop)
  predict()      return predictions
  evaluate()     compute RMSE, R² + save plots
  save() / load() persist model to disk

  ※ save(use_pipeline=...) : 학습 시 heterogeneity pipeline 사용 여부를
     pt 파일에 함께 기록. predict.py가 이 값을 읽어서
     예측 시에도 동일하게 pipeline을 적용할지 자동 판단.
     (torch.save도 결국 딕셔너리를 저장하는 것이므로 pkl과 동일한 방식으로
      use_pipeline 키를 추가하면 됨)
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
        # ── heterogeneity pipeline 사용 여부 (기본 False) ──
        # train_static()에서 use_pipeline=True로 학습되면 True로 설정되고
        # save() 시 pt 파일에 함께 저장됨. predict.py가 load() 후 이 값을 읽어서
        # 9개 raw 입력값을 230차원으로 변환할지 판단하는 데 사용.
        self.use_pipeline = False

    def train(self, X_train, y_train, x_cols=None, scaler=None):
        """
        X_train : (n_train, 9) or (n_train, 230)  already scaled
                  9   = raw concentration features (pipeline off)
                  230 = media representation vector (pipeline on)
        y_train : (n_train,)
        x_cols  : feature names
                  ※ pipeline on이어도 x_cols는 원본 9개 컴포넌트 이름 그대로 유지됨
                    (get_static_data가 x_cols를 raw 컬럼명으로 반환하기 때문)
        scaler  : fitted StandardScaler (saved for inference)
                  ※ pipeline on이면 230차원 기준으로 fit된 scaler가 들어옴
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

        # ── input_dim은 X_train.shape[1]로 자동 결정 ──
        # pipeline off면 9, pipeline on이면 230 → 네트워크 첫 레이어가 자동으로 맞춰짐
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

        # ── 학습 정보 기록용: 학습 후에만 확정되는 값 저장 ──
        self.best_epoch = best_epoch
        self.best_val   = round(float(best_val), 4)

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

    def get_config(self):
        """
        학습 정보 기록용 하이퍼파라미터 리포트.
        train.py의 train_model()이 result.json의 meta.hyperparams에
        이 값을 그대로 저장함.

        ※ epoch/lr/hidden_dims/dropout/batch_size는 config.py에서 미리
          고정된 값. best_epoch/best_val은 train()이 끝난 뒤에만
          알 수 있는 값이라 getattr로 방어적으로 조회함
          (evaluate() 이전에 호출되는 경우는 없지만 혹시 몰라 대비).
        """
        return {
            "epoch"       : config.MLP_EPOCHS,
            "hidden_dims" : config.MLP_HIDDEN_DIMS,
            "dropout"     : config.MLP_DROPOUT,
            "lr"          : config.MLP_LR,
            "batch_size"  : config.MLP_BATCH_SIZE,
            "best_epoch"  : getattr(self, "best_epoch", None),
            "best_val"    : getattr(self, "best_val", None),
        }

    def save(self, use_pipeline: bool = False):
        """
        모델을 pt로 저장.

        use_pipeline : 학습 시 heterogeneity pipeline(SMILES·RDKit·GEM)
                       사용 여부. train.py의 train_static()에서 전달됨.
                       이 값이 pt 파일 안에 함께 저장되어, predict.py가
                       load() 후 동일한 전처리를 자동으로 재현할 수 있게 함.
                       (torch.save도 dict를 저장하는 것이라 pkl과 동일한
                        방식으로 키를 추가하면 됨)
        """
        self.use_pipeline = use_pipeline
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict"  : self.net.state_dict(),
            "y_scaler"    : self.y_scaler,
            "scaler"      : self.scaler,
            "x_cols"      : self.x_cols,
            "input_dim"   : next(self.net.parameters()).shape[1],
            # ── 예측 시 pipeline 적용 여부 판단용 ──
            "use_pipeline": self.use_pipeline,
        }, SAVE_PATH)
        print(f"[MLP] Saved: {SAVE_PATH}  (use_pipeline={self.use_pipeline})")

    def load(self):
        ckpt = torch.load(SAVE_PATH, map_location=self.device, weights_only=False)
        self.net = _MLPNet(
            ckpt["input_dim"], config.MLP_HIDDEN_DIMS, config.MLP_DROPOUT
        ).to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        self.y_scaler = ckpt["y_scaler"]
        self.scaler   = ckpt["scaler"]
        self.x_cols   = ckpt["x_cols"]
        # ── 구버전 pt 호환: use_pipeline 키가 없으면 False로 간주 ──
        self.use_pipeline = ckpt.get("use_pipeline", False)
        print(f"[MLP] Loaded: {SAVE_PATH}  (use_pipeline={self.use_pipeline})")


if __name__ == "__main__":
    from data_preprocess import get_static_data
    X_train, X_test, y_train, y_test, x_cols, scaler = get_static_data()

    model = MLPModel()
    model.train(X_train, y_train, x_cols=x_cols, scaler=scaler)
    model.evaluate(X_test, y_test)
    model.save()