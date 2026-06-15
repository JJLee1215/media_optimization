"""
process3_1_evaluate.py
학습된 모델로:
  1. titer 예측값 vs 실제값 scatter plot
  2. 학습된 adjacency 행렬 Ã 히트맵
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # 서버 환경 (GUI 없음)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from process3_1_config import Config
from process3_1_dataset import get_dataloaders
from process3_1_model import Model3

# ── 경로 설정 ──────────────────────────────
MODEL_PATH  = "/app/models/model3_best.pt"
OUTPUT_DIR  = "/app/outputs/model3"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@torch.no_grad()
def collect_predictions(model, loader, device):
    """전체 데이터셋 예측값 수집"""
    model.eval()
    all_true_titer, all_pred_titer = [], []
    all_true_viab,  all_pred_viab  = [], []
    all_A_tilde = []

    for m_static, X_dynamic, y_titer, y_viab in loader:
        m_static  = m_static.to(device)
        X_dynamic = X_dynamic.to(device)

        mu_titer, pred_viab, A_tilde = model(m_static, X_dynamic)

        all_true_titer.append(y_titer.numpy())
        all_pred_titer.append(mu_titer.cpu().numpy())
        all_true_viab.append(y_viab.numpy())
        all_pred_viab.append(pred_viab.cpu().numpy())
        all_A_tilde.append(A_tilde.cpu().numpy())

    return (
        np.concatenate(all_true_titer),
        np.concatenate(all_pred_titer),
        np.concatenate(all_true_viab),
        np.concatenate(all_pred_viab),
        np.concatenate(all_A_tilde, axis=0).mean(axis=0),  # 배치 평균 A
    )


def plot_prediction(true_vals, pred_vals, title, save_path):
    """예측값 vs 실제값 scatter plot"""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(true_vals, pred_vals, color="#1D9E75", alpha=0.8,
               edgecolors="white", s=80, zorder=3, label="Predicted")

    # 이상적인 예측선 (y=x)
    vmin = min(true_vals.min(), pred_vals.min()) * 0.95
    vmax = max(true_vals.max(), pred_vals.max()) * 1.05
    ax.plot([vmin, vmax], [vmin, vmax], "--", color="#888780",
            linewidth=1.2, label="Ideal (y=x)")

    # RMSE
    rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
    ax.text(0.05, 0.93, f"RMSE = {rmse:.4f}",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F1EFE8", alpha=0.8))

    ax.set_xlabel("Actual", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"저장: {save_path}")


def plot_adjacency(A_mean, variable_names, save_path):
    """학습된 Ã 행렬 히트맵"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── 왼쪽: 학습된 Ã ──────────────────────
    sns.heatmap(
        A_mean,
        ax=axes[0],
        xticklabels=variable_names,
        yticklabels=variable_names,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        annot=True, fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Connection Strength"}
    )
    axes[0].set_title("Learned Adjacency (Ã)", fontsize=13, fontweight="bold", pad=10)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    # ── 오른쪽: A₀ (domain prior) ────────────
    cfg = Config()
    A0_np = cfg.A0.numpy()

    sns.heatmap(
        A0_np,
        ax=axes[1],
        xticklabels=variable_names,
        yticklabels=variable_names,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        annot=True, fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Connection Strength"}
    )
    axes[1].set_title("A0 (Domain Prior)", fontsize=13, fontweight="bold", pad=10)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.suptitle("Learned Graph vs Domain Prior", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"저장: {save_path}")


def plot_loss_curves(history, save_path):
    """학습 곡선"""
    epochs = range(1, len(history["train"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # train / val loss
    axes[0].plot(epochs, history["train"], color="#1D9E75", label="train")
    axes[0].plot(epochs, history["val"],   color="#534AB7", label="val", linestyle="--")
    axes[0].set_title("Train / Val Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 항목별 loss
    axes[1].plot(epochs, history["titer"], color="#1D9E75", label="L_titer")
    axes[1].plot(epochs, history["viab"],  color="#534AB7", label="L_viab")
    axes[1].plot(epochs, history["graph"], color="#E24B4A", label="L_graph")
    axes[1].set_title("Loss by Component", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"저장: {save_path}")


def evaluate(cfg: Config, history=None):
    device = cfg.device

    # ── 모델 로드 ────────────────────────────
    model = Model3(cfg).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"모델 로드: {MODEL_PATH}")

    # ── 전체 데이터로 예측 ────────────────────
    train_loader, val_loader = get_dataloaders(cfg)

    # train + val 합쳐서 시각화
    from torch.utils.data import ConcatDataset, DataLoader
    full_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
    full_loader  = DataLoader(full_dataset, batch_size=cfg.batch_size)

    true_titer, pred_titer, true_viab, pred_viab, A_mean = collect_predictions(
        model, full_loader, device
    )

    print(f"\n전체 샘플 수: {len(true_titer)}")
    print(f"Titer  RMSE: {np.sqrt(np.mean((true_titer - pred_titer)**2)):.4f}")
    print(f"Viab   RMSE: {np.sqrt(np.mean((true_viab  - pred_viab )**2)):.4f}")

    # ── 시각화 ───────────────────────────────
    # 1. Titer 예측 scatter
    plot_prediction(
        true_titer, pred_titer,
        title="Titer: Predicted vs Actual",
        save_path=os.path.join(OUTPUT_DIR, "titer_prediction.png")
    )

    # 2. Viability 예측 scatter
    plot_prediction(
        true_viab, pred_viab,
        title="Viability: Predicted vs Actual",
        save_path=os.path.join(OUTPUT_DIR, "viab_prediction.png")
    )

    # 3. Adjacency 히트맵
    plot_adjacency(
        A_mean, cfg.variable_names,
        save_path=os.path.join(OUTPUT_DIR, "adjacency_heatmap.png")
    )

    # 4. Loss 곡선 (history가 있을 때만)
    if history is not None:
        plot_loss_curves(
            history,
            save_path=os.path.join(OUTPUT_DIR, "loss_curves.png")
        )

    print(f"\n모든 시각화 저장 완료: {OUTPUT_DIR}/")


if __name__ == "__main__":
    cfg = Config()

    # train 후 바로 evaluate
    from process3_1_train import train
    model, val_loader, history = train(cfg)
    evaluate(cfg, history=history)