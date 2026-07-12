"""
train_module.py
StaticTimeGNN training / evaluation wrapper

Methods:
  train()    fit model
  evaluate() compute metrics + save plots
  save()     persist to disk
  load()     restore from disk
"""

import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from .model import StaticTimeGNN
from .loss  import StaticTimeGNNLoss

MODEL_NAME = "static_time_gnn"
SAVE_PATH  = config.model_save_path(MODEL_NAME)
RESULT_DIR = config.result_dir(MODEL_NAME)


class StaticTimeGNNModel:
    def __init__(self, d_static: int, d_dynamic: int, N: int, A0: torch.Tensor):
        """
        d_static  : m_static feature count
        d_dynamic : X_dynamic feature count
        N         : graph node count (= d_dynamic)
        A0        : (N, N) domain prior
        """
        self.device  = config.DEVICE
        self.net     = StaticTimeGNN(d_static, d_dynamic, N, A0).to(self.device)
        self.loss_fn = StaticTimeGNNLoss(
            A0           = A0,
            lambda_viab  = config.GNN_LAMBDA_VIAB,
            lambda_graph = config.GNN_LAMBDA_GRAPH,
            huber_delta  = config.GNN_HUBER_DELTA,
        ).to(self.device)
        self.history = {"train": [], "val": [], "titer": [], "viab": [], "graph": []}

        # ── 학습 정보 기록용: 생성자 파라미터 저장 ──
        # get_config()에서 모델 구조 정보(d_static, d_dynamic, N)를
        # result.json의 meta.hyperparams에 함께 기록하기 위해 보관.
        self.d_static  = d_static
        self.d_dynamic = d_dynamic
        self.N         = N

    def _run_epoch(self, loader, optimizer=None):
        training = optimizer is not None
        self.net.train() if training else self.net.eval()
        sums = {"total": 0, "titer": 0, "viab": 0, "graph": 0}
        ctx  = torch.enable_grad() if training else torch.no_grad()

        with ctx:
            for m, X, yt, yv in loader:
                m, X, yt, yv = (t.to(self.device) for t in (m, X, yt, yv))
                mu, viab, A  = self.net(m, X)
                loss, d      = self.loss_fn(mu, viab, A, yt, yv)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                for k in sums:
                    sums[k] += d[k]

        n = len(loader)
        return {k: v / n for k, v in sums.items()}

    def train(self, train_loader, val_loader):
        optimizer = optim.Adam(self.net.parameters(), lr=config.GNN_LR)
        best_val, best_epoch = float("inf"), 0

        n_params = sum(p.numel() for p in self.net.parameters())
        print(f"[StaticTimeGNN] Training...  params={n_params:,}")
        t0 = time.time()

        for epoch in range(1, config.GNN_EPOCHS + 1):
            tr = self._run_epoch(train_loader, optimizer)
            vl = self._run_epoch(val_loader)

            self.history["train"].append(tr["total"])
            self.history["val"].append(vl["total"])
            self.history["titer"].append(tr["titer"])
            self.history["viab"].append(tr["viab"])
            self.history["graph"].append(tr["graph"])

            if vl["total"] < best_val:
                best_val, best_epoch = vl["total"], epoch
                torch.save(self.net.state_dict(), SAVE_PATH.parent / "stgnn_tmp.pt")

            if epoch % 10 == 0 or epoch == 1:
                print(f"  [{epoch:>3}/{config.GNN_EPOCHS}] "
                      f"train={tr['total']:.4f} "
                      f"(t={tr['titer']:.4f} v={tr['viab']:.4f} g={tr['graph']:.4f})  "
                      f"val={vl['total']:.4f}")

        self.net.load_state_dict(
            torch.load(SAVE_PATH.parent / "stgnn_tmp.pt", weights_only=True)
        )
        print(f"[StaticTimeGNN] Training complete.  {time.time()-t0:.1f}s  best val: {best_val:.4f}@{best_epoch}")

        # ── 학습 정보 기록용: 학습 후에만 확정되는 값 저장 ──
        self.best_epoch = best_epoch
        self.best_val   = round(float(best_val), 4)

    @torch.no_grad()
    def predict(self, m, X):
        self.net.eval()
        mu, viab, A = self.net(m.to(self.device), X.to(self.device))
        return mu.cpu().numpy(), viab.cpu().numpy(), A.cpu().numpy()

    def evaluate(self, train_loader, val_loader):
        full_loader = DataLoader(
            ConcatDataset([train_loader.dataset, val_loader.dataset]),
            batch_size=config.GNN_BATCH_SIZE,
        )

        all_yt, all_mu, all_yv, all_viab, all_A = [], [], [], [], []
        for m, X, yt, yv in full_loader:
            mu, viab, A = self.predict(m, X)
            all_yt.append(yt.numpy()); all_mu.append(mu)
            all_yv.append(yv.numpy()); all_viab.append(viab)
            all_A.append(A)

        y_titer = np.concatenate(all_yt);  p_titer = np.concatenate(all_mu)
        y_viab  = np.concatenate(all_yv);  p_viab  = np.concatenate(all_viab)
        A_mean  = np.concatenate(all_A, axis=0).mean(axis=0)

        rmse_t = np.sqrt(mean_squared_error(y_titer, p_titer))
        rmse_v = np.sqrt(mean_squared_error(y_viab,  p_viab))
        print(f"[StaticTimeGNN] Titer RMSE: {rmse_t:.4f}  Viab RMSE: {rmse_v:.4f}")

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        self._plot_prediction(y_titer, p_titer, "Titer: Predicted vs Actual",
                              RESULT_DIR / "titer_prediction.png")
        self._plot_prediction(y_viab,  p_viab,  "Viability: Predicted vs Actual",
                              RESULT_DIR / "viab_prediction.png")
        self._plot_adjacency(A_mean)
        self._plot_loss_curves()

        return {
            "model"      : MODEL_NAME,
            "titer_rmse" : round(float(rmse_t), 4),
            "viab_rmse"  : round(float(rmse_v), 4),
        }

    def _plot_prediction(self, y_true, y_pred, title, path):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, color="#1D9E75", alpha=0.8, edgecolors="white", s=80)
        vmin = min(y_true.min(), y_pred.min()) * 0.95
        vmax = max(y_true.max(), y_pred.max()) * 1.05
        ax.plot([vmin, vmax], [vmin, vmax], "--", color="#888780", lw=1.2, label="Ideal (y=x)")
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        ax.text(0.05, 0.93, f"RMSE={rmse:.4f}", transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F1EFE8", alpha=0.8))
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        print(f"[StaticTimeGNN] Saved: {path}")

    def _plot_adjacency(self, A_mean):
        var_names = ["var_" + str(i) for i in range(A_mean.shape[0])]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(A_mean, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
                    annot=(A_mean.shape[0] <= 15), fmt=".2f", linewidths=0.5,
                    cbar_kws={"label": "Connection Strength"})
        ax.set_title("Learned Adjacency (Ã)", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        path = RESULT_DIR / "adjacency_heatmap.png"
        plt.savefig(path, dpi=150); plt.close()
        print(f"[StaticTimeGNN] Saved: {path}")

    def _plot_loss_curves(self):
        epochs = range(1, len(self.history["train"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, self.history["train"], color="#1D9E75", label="train")
        axes[0].plot(epochs, self.history["val"],   color="#534AB7", label="val", ls="--")
        axes[0].set_title("Train / Val Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(epochs, self.history["titer"], color="#1D9E75", label="L_titer")
        axes[1].plot(epochs, self.history["viab"],  color="#534AB7", label="L_viab")
        axes[1].plot(epochs, self.history["graph"], color="#E24B4A", label="L_graph")
        axes[1].set_title("Loss by Component"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        path = RESULT_DIR / "loss_curves.png"
        plt.savefig(path, dpi=150); plt.close()
        print(f"[StaticTimeGNN] Saved: {path}")

    def get_config(self):
        """
        학습 정보 기록용 하이퍼파라미터 리포트.
        train.py의 train_model()이 result.json의 meta.hyperparams에
        이 값을 그대로 저장함.

        ※ d_static/d_dynamic/N은 생성자에서 받은 모델 구조 정보(정적으로 확정).
          lr/epoch/lambda_viab/lambda_graph/huber_delta는 config.py에서
          미리 고정된 값. best_epoch/best_val은 train()이 끝난 뒤에만
          알 수 있는 값이라 getattr로 방어적으로 조회함.
        """
        return {
            "epoch"        : config.GNN_EPOCHS,
            "lr"           : config.GNN_LR,
            "d_static"     : self.d_static,
            "d_dynamic"    : self.d_dynamic,
            "N"            : self.N,
            "lambda_viab"  : config.GNN_LAMBDA_VIAB,
            "lambda_graph" : config.GNN_LAMBDA_GRAPH,
            "huber_delta"  : config.GNN_HUBER_DELTA,
            "best_epoch"   : getattr(self, "best_epoch", None),
            "best_val"     : getattr(self, "best_val", None),
        }

    def save(self):
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), SAVE_PATH)
        print(f"[StaticTimeGNN] Saved: {SAVE_PATH}")

    def load(self):
        self.net.load_state_dict(
            torch.load(SAVE_PATH, map_location=self.device, weights_only=True)
        )
        self.net = self.net.to(self.device)
        print(f"[StaticTimeGNN] Loaded: {SAVE_PATH}")