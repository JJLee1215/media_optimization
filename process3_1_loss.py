import torch
import torch.nn as nn
from process3_1_config import Config


class Model3Loss(nn.Module):
    """
    L = L_titer + λ₁·L_viab + λ₂·L_graph

    L_titer  = MSE(mu_titer, y_true_titer)
    L_viab   = Huber(y_viab, y_true_viab)
    L_graph  = ||A_tilde - A₀||²_F
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.lambda_viab  = cfg.lambda_viab
        self.lambda_graph = cfg.lambda_graph

        self.mse   = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=cfg.huber_delta)

        # A₀ (고정)
        self.register_buffer("A0", cfg.A0)

    def forward(self, mu_titer, y_viab, A_tilde, y_true_titer, y_true_viab):
        """
        입력:
            mu_titer      (batch,)    titer 예측값
            y_viab        (batch,)    viability 예측값
            A_tilde       (batch, N, N)  학습된 adjacency
            y_true_titer  (batch,)    titer 정답
            y_true_viab   (batch,)    viability 정답
        출력:
            loss          scalar
            loss_dict     각 항목별 loss 값 (로깅용)
        """
        # ── L_titer ──────────────────────────────
        L_titer = self.mse(mu_titer, y_true_titer)

        # ── L_viab ───────────────────────────────
        L_viab = self.huber(y_viab, y_true_viab)

        # ── L_graph ──────────────────────────────
        # ||A_tilde - A₀||²_F  (Frobenius norm)
        A0_expanded = self.A0.unsqueeze(0).expand_as(A_tilde)
        L_graph = torch.mean((A_tilde - A0_expanded) ** 2)

        # ── 합산 ─────────────────────────────────
        loss = L_titer + self.lambda_viab * L_viab + self.lambda_graph * L_graph

        loss_dict = {
            "loss_total" : loss.item(),
            "loss_titer" : L_titer.item(),
            "loss_viab"  : L_viab.item(),
            "loss_graph" : L_graph.item(),
        }

        return loss, loss_dict


if __name__ == "__main__":
    cfg = Config()
    criterion = Model3Loss(cfg)

    batch = 4
    N = cfg.N

    mu_titer     = torch.randn(batch)
    y_viab       = torch.rand(batch)
    A_tilde      = torch.rand(batch, N, N)
    y_true_titer = torch.randn(batch)
    y_true_viab  = torch.rand(batch)

    loss, loss_dict = criterion(mu_titer, y_viab, A_tilde, y_true_titer, y_true_viab)
    print(f"loss_total : {loss_dict['loss_total']:.4f}")
    print(f"loss_titer : {loss_dict['loss_titer']:.4f}")
    print(f"loss_viab  : {loss_dict['loss_viab']:.4f}")
    print(f"loss_graph : {loss_dict['loss_graph']:.4f}")