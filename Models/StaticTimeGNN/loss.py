"""
loss.py
StaticTimeGNN Loss Function

L = L_titer + λ_viab · L_viab + λ_graph · L_graph

L_titer  = MSE(mu_titer, y_true_titer)
L_viab   = Huber(y_viab, y_true_viab)    λ_viab  = 0.5
L_graph  = ||Ã - A0||²_F                 λ_graph = 1.0
"""

import torch
import torch.nn as nn


class StaticTimeGNNLoss(nn.Module):
    def __init__(self, A0: torch.Tensor, lambda_viab: float, lambda_graph: float, huber_delta: float):
        super().__init__()
        self.lambda_viab  = lambda_viab
        self.lambda_graph = lambda_graph
        self.mse          = nn.MSELoss()
        self.huber        = nn.HuberLoss(delta=huber_delta)
        self.register_buffer("A0", A0)

    def forward(self, mu_titer, y_viab, A_tilde, y_true_titer, y_true_viab):
        """
        mu_titer     : (batch,)      predicted titer
        y_viab       : (batch,)      predicted viability
        A_tilde      : (batch, N, N) learned adjacency
        y_true_titer : (batch,)
        y_true_viab  : (batch,)

        Returns:
          loss      scalar
          loss_dict {"total", "titer", "viab", "graph"}
        """
        L_titer = self.mse(mu_titer, y_true_titer)
        L_viab  = self.huber(y_viab, y_true_viab)
        L_graph = torch.mean((A_tilde - self.A0.unsqueeze(0).expand_as(A_tilde)) ** 2)

        loss = L_titer + self.lambda_viab * L_viab + self.lambda_graph * L_graph

        return loss, {
            "total" : loss.item(),
            "titer" : L_titer.item(),
            "viab"  : L_viab.item(),
            "graph" : L_graph.item(),
        }