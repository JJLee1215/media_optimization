import os
import torch
import torch.optim as optim
from process3_1_config import Config
from process3_1_dataset import get_dataloaders
from process3_1_model import Model3
from process3_1_loss import Model3Loss

# 모델 저장 경로
MODEL_SAVE_PATH = "/app/models/model3_best.pt"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    loss_sums  = {"loss_titer": 0, "loss_viab": 0, "loss_graph": 0}

    for m_static, X_dynamic, y_titer, y_viab in loader:
        m_static  = m_static.to(device)
        X_dynamic = X_dynamic.to(device)
        y_titer   = y_titer.to(device)
        y_viab    = y_viab.to(device)

        optimizer.zero_grad()
        mu_titer, pred_viab, A_tilde = model(m_static, X_dynamic)
        loss, loss_dict = criterion(mu_titer, pred_viab, A_tilde, y_titer, y_viab)
        loss.backward()
        optimizer.step()

        total_loss += loss_dict["loss_total"]
        for k in loss_sums:
            loss_sums[k] += loss_dict[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in loss_sums.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    loss_sums  = {"loss_titer": 0, "loss_viab": 0, "loss_graph": 0}

    for m_static, X_dynamic, y_titer, y_viab in loader:
        m_static  = m_static.to(device)
        X_dynamic = X_dynamic.to(device)
        y_titer   = y_titer.to(device)
        y_viab    = y_viab.to(device)

        mu_titer, pred_viab, A_tilde = model(m_static, X_dynamic)
        loss, loss_dict = criterion(mu_titer, pred_viab, A_tilde, y_titer, y_viab)

        total_loss += loss_dict["loss_total"]
        for k in loss_sums:
            loss_sums[k] += loss_dict[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in loss_sums.items()}


def train(cfg: Config):
    device = cfg.device
    print(f"device: {device}")

    # ── /app/models 폴더 없으면 생성 ──────────
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # ── 데이터 ───────────────────────────────
    train_loader, val_loader = get_dataloaders(cfg)
    print(f"train: {len(train_loader.dataset)}개  val: {len(val_loader.dataset)}개")

    # ── 모델 / Loss / Optimizer ──────────────
    model     = Model3(cfg).to(device)
    criterion = Model3Loss(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 수: {total_params:,}\n")

    # ── 학습 루프 ────────────────────────────
    best_val_loss = float("inf")
    history = {"train": [], "val": [], "titer": [], "viab": [], "graph": []}

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_dict = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_dict   = validate(model, val_loader, criterion, device)

        # 히스토리 저장 (시각화용)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["titer"].append(train_dict["loss_titer"])
        history["viab"].append(train_dict["loss_viab"])
        history["graph"].append(train_dict["loss_graph"])

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[Epoch {epoch:3d}/{cfg.epochs}] "
                f"train: {train_loss:.4f} "
                f"(titer={train_dict['loss_titer']:.4f} "
                f"viab={train_dict['loss_viab']:.4f} "
                f"graph={train_dict['loss_graph']:.4f})  "
                f"val: {val_loss:.4f}"
            )

        # best model 저장 → /app/models/model3_best.pt
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\n학습 완료. best val loss: {best_val_loss:.4f}")
    print(f"모델 저장 위치: {MODEL_SAVE_PATH}")
    return model, val_loader, history


if __name__ == "__main__":
    cfg = Config()
    model, val_loader, history = train(cfg)