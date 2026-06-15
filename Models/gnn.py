"""
Models/Model_GNN.py
GNN (Model3) — 배지 조성(정적) + 시계열 공정변수

구조:
  StaticEncoder   → h0, c0
  DynamicEncoder  → H_dynamic
  CrossAttention  → H_dynamic*
  GraphConstruct  → V, Ã
  GNN             → H_final
  OutputHead      → μ_titer, ŷ_viab
"""

import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config

cfg = Config()
MODEL_NAME = "gnn"
SAVE_PATH  = cfg.model_save_path(MODEL_NAME)
RESULT_DIR = cfg.result_dir(MODEL_NAME)


# ── 내부 신경망 컴포넌트 ──────────────────────────────

class StaticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.h0 = nn.Sequential(nn.Linear(cfg.GNN_D_STATIC, cfg.GNN_D_HIDDEN), nn.Tanh())
        self.c0 = nn.Sequential(nn.Linear(cfg.GNN_D_STATIC, cfg.GNN_D_HIDDEN), nn.Tanh())

    def forward(self, m):
        return self.h0(m).unsqueeze(0), self.c0(m).unsqueeze(0)


class DynamicEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(cfg.GNN_D_DYNAMIC, cfg.GNN_D_HIDDEN, batch_first=True)

    def forward(self, X, h0, c0):
        H, _ = self.lstm(X, (h0, c0))
        return H


class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(cfg.GNN_D_HIDDEN, cfg.GNN_D_HIDDEN, bias=False)
        self.Wk = nn.Linear(cfg.GNN_D_STATIC,  cfg.GNN_D_HIDDEN, bias=False)
        self.Wv = nn.Linear(cfg.GNN_D_STATIC,  cfg.GNN_D_HIDDEN, bias=False)
        self.norm  = nn.LayerNorm(cfg.GNN_D_HIDDEN)
        self.scale = cfg.GNN_D_HIDDEN ** 0.5

    def forward(self, H, m):
        Q    = self.Wq(H)
        K    = self.Wk(m).unsqueeze(1)
        V    = self.Wv(m).unsqueeze(1)
        attn = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        return self.norm(H + torch.bmm(attn, V))


class GraphConstruction(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wa = nn.Linear(cfg.GNN_D_HIDDEN, cfg.GNN_D_HIDDEN, bias=False)
        self.register_buffer("A0",       cfg.GNN_A0)
        self.register_buffer("M_causal", torch.ones(cfg.GNN_N, cfg.GNN_N))

    def forward(self, H_star):
        V      = H_star[:, -1, :].unsqueeze(1).expand(-1, cfg.GNN_N, -1)
        V_proj = self.Wa(V)
        A      = torch.sigmoid(torch.bmm(V_proj, V_proj.transpose(1, 2)))
        A_tilde = (A * self.M_causal + self.A0.unsqueeze(0)).clamp(0, 1)
        return V, A_tilde


class GNNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(cfg.GNN_D_HIDDEN, cfg.GNN_D_HIDDEN)

    def forward(self, H, A):
        deg    = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        A_norm = A / deg
        return F.relu(self.W(torch.bmm(A_norm, H)))


class OutputHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.titer = nn.Sequential(
            nn.Linear(cfg.GNN_D_HIDDEN, cfg.GNN_MLP_HIDDEN), nn.ReLU(),
            nn.Linear(cfg.GNN_MLP_HIDDEN, 1)
        )
        self.viab = nn.Sequential(
            nn.Linear(cfg.GNN_D_HIDDEN, cfg.GNN_MLP_HIDDEN), nn.ReLU(),
            nn.Linear(cfg.GNN_MLP_HIDDEN, 1), nn.Sigmoid()
        )

    def forward(self, H_final):
        h = H_final.mean(dim=1)
        return self.titer(h).squeeze(-1), self.viab(h).squeeze(-1)


class _Model3Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.static_enc  = StaticEncoder()
        self.dynamic_enc = DynamicEncoder()
        self.cross_attn  = CrossAttention()
        self.graph_build = GraphConstruction()
        self.gnn_layers  = nn.ModuleList([GNNLayer() for _ in range(cfg.GNN_N_LAYERS)])
        self.output_head = OutputHead()

    def forward(self, m, X):
        h0, c0   = self.static_enc(m)
        H        = self.dynamic_enc(X, h0, c0)
        H_star   = self.cross_attn(H, m)
        V, A     = self.graph_build(H_star)
        for layer in self.gnn_layers:
            V = layer(V, A)
        mu, viab = self.output_head(V)
        return mu, viab, A


# ── Loss ──────────────────────────────────────────────

class _GNNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse   = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=cfg.GNN_HUBER_DELTA)
        self.register_buffer("A0", cfg.GNN_A0)

    def forward(self, mu, viab, A, y_titer, y_viab):
        L_titer = self.mse(mu, y_titer)
        L_viab  = self.huber(viab, y_viab)
        L_graph = torch.mean((A - self.A0.unsqueeze(0).expand_as(A)) ** 2)
        loss    = L_titer + cfg.GNN_LAMBDA_VIAB * L_viab + cfg.GNN_LAMBDA_GRAPH * L_graph
        return loss, {
            "total": loss.item(), "titer": L_titer.item(),
            "viab": L_viab.item(), "graph": L_graph.item()
        }


# ── GNNModel 래퍼 ────────────────────────────────────

class GNNModel:
    def __init__(self):
        self.net     = _Model3Net()
        self.loss_fn = _GNNLoss()
        self.device  = cfg.DEVICE
        self.history = {"train": [], "val": [], "titer": [], "viab": [], "graph": []}

    def _run_epoch(self, loader, optimizer=None):
        training = optimizer is not None
        self.net.train() if training else self.net.eval()
        sums = {"total": 0, "titer": 0, "viab": 0, "graph": 0}
        ctx  = torch.enable_grad() if training else torch.no_grad()

        with ctx:
            for m, X, yt, yv in loader:
                m, X  = m.to(self.device), X.to(self.device)
                yt, yv = yt.to(self.device), yv.to(self.device)
                mu, viab, A = self.net(m, X)
                loss, d = self.loss_fn(mu, viab, A, yt, yv)
                if training:
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                for k in sums: sums[k] += d[k]

        n = len(loader)
        return {k: v / n for k, v in sums.items()}

    def train(self, train_loader, val_loader):
        self.net = self.net.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)
        optimizer = optim.Adam(self.net.parameters(), lr=cfg.GNN_LR)
        best_val, best_epoch = float("inf"), 0

        print(f"[GNN] 학습 시작  파라미터: {sum(p.numel() for p in self.net.parameters()):,}")
        t0 = time.time()

        for epoch in range(1, cfg.GNN_EPOCHS + 1):
            tr = self._run_epoch(train_loader, optimizer)
            vl = self._run_epoch(val_loader)

            self.history["train"].append(tr["total"])
            self.history["val"].append(vl["total"])
            self.history["titer"].append(tr["titer"])
            self.history["viab"].append(tr["viab"])
            self.history["graph"].append(tr["graph"])

            if vl["total"] < best_val:
                best_val, best_epoch = vl["total"], epoch
                torch.save(self.net.state_dict(), SAVE_PATH.parent / "gnn_tmp.pt")

            if epoch % 10 == 0 or epoch == 1:
                print(f"  [{epoch:>3}/{cfg.GNN_EPOCHS}] "
                      f"train={tr['total']:.4f} "
                      f"(t={tr['titer']:.4f} v={tr['viab']:.4f} g={tr['graph']:.4f})  "
                      f"val={vl['total']:.4f}")

        self.net.load_state_dict(torch.load(SAVE_PATH.parent / "gnn_tmp.pt", weights_only=True))
        print(f"[GNN] 완료  {time.time()-t0:.1f}초  best val: {best_val:.4f}@{best_epoch}")

    @torch.no_grad()
    def predict(self, m, X):
        self.net.eval()
        mu, viab, A = self.net(
            m.to(self.device), X.to(self.device)
        )
        return mu.cpu().numpy(), viab.cpu().numpy(), A.cpu().numpy()

    def evaluate(self, train_loader, val_loader):
        from torch.utils.data import ConcatDataset, DataLoader
        full = DataLoader(
            ConcatDataset([train_loader.dataset, val_loader.dataset]),
            batch_size=cfg.GNN_BATCH_SIZE
        )

        all_yt, all_mu, all_yv, all_viab, all_A = [], [], [], [], []
        for m, X, yt, yv in full:
            mu, viab, A = self.predict(m, X)
            all_yt.append(yt.numpy()); all_mu.append(mu)
            all_yv.append(yv.numpy()); all_viab.append(viab)
            all_A.append(A)

        y_titer  = np.concatenate(all_yt);  p_titer = np.concatenate(all_mu)
        y_viab   = np.concatenate(all_yv);  p_viab  = np.concatenate(all_viab)
        A_mean   = np.concatenate(all_A, axis=0).mean(axis=0)

        rmse_t = np.sqrt(mean_squared_error(y_titer, p_titer))
        rmse_v = np.sqrt(mean_squared_error(y_viab,  p_viab))
        print(f"[GNN] Titer RMSE: {rmse_t:.4f}  Viab RMSE: {rmse_v:.4f}")

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        self._plot_prediction(y_titer, p_titer, "Titer: Predicted vs Actual", RESULT_DIR / "titer_prediction.png")
        self._plot_prediction(y_viab,  p_viab,  "Viab: Predicted vs Actual",  RESULT_DIR / "viab_prediction.png")
        self._plot_adjacency(A_mean)
        self._plot_loss_curves()

        return {"model": MODEL_NAME,
                "titer_rmse": round(float(rmse_t), 4),
                "viab_rmse":  round(float(rmse_v), 4)}

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

    def _plot_adjacency(self, A_mean):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        kw = dict(xticklabels=cfg.GNN_VARIABLE_NAMES, yticklabels=cfg.GNN_VARIABLE_NAMES,
                  cmap="YlOrRd", vmin=0, vmax=1, annot=True, fmt=".2f",
                  linewidths=0.5, cbar_kws={"label": "Connection Strength"})
        sns.heatmap(A_mean,          ax=axes[0], **kw); axes[0].set_title("Learned Ã")
        sns.heatmap(cfg.GNN_A0.numpy(), ax=axes[1], **kw); axes[1].set_title("A0 (Domain Prior)")
        for ax in axes:
            ax.tick_params(axis="x", rotation=45)
        plt.suptitle("Learned Graph vs Domain Prior", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "adjacency_heatmap.png", dpi=150)
        plt.close()

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
        plt.savefig(RESULT_DIR / "loss_curves.png", dpi=150)
        plt.close()

    def save(self):
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), SAVE_PATH)
        print(f"[GNN] 저장: {SAVE_PATH}")

    def load(self):
        self.net.load_state_dict(torch.load(SAVE_PATH, map_location=self.device, weights_only=True))
        self.net = self.net.to(self.device)
        print(f"[GNN] 로드: {SAVE_PATH}")


if __name__ == "__main__":
    from data import get_gnn_data
    train_loader, val_loader = get_gnn_data(use_syn=True)
    model = GNNModel()
    model.train(train_loader, val_loader)
    model.evaluate(train_loader, val_loader)
    model.save()