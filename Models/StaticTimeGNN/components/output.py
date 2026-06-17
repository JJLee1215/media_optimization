"""
components/output.py
Output Head

Mean pooling → two MLP heads

Input  : H_final (batch, N, d_hidden)  from GNN
Output : mu_titer (batch,)   predicted titer
         y_viab   (batch,)   predicted viability (0~1)

Steps:
  1. Mean pooling: (batch, N, d_hidden) → (batch, d_hidden)
  2. MLP_titer: (batch, d_hidden) → (batch, 1) → squeeze
  3. MLP_viab:  (batch, d_hidden) → (batch, 1) → sigmoid → squeeze
"""

import torch
import torch.nn as nn


class OutputHead(nn.Module):
    def __init__(self, d_hidden: int, mlp_hidden: int):
        super().__init__()
        self.mlp_titer = nn.Sequential(
            nn.Linear(d_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )
        self.mlp_viab = nn.Sequential(
            nn.Linear(d_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, H_final: torch.Tensor):
        """
        H_final  : (batch, N, d_hidden)
        Returns  : mu_titer (batch,)
                   y_viab   (batch,)
        """
        h_pool   = H_final.mean(dim=1)                    # (batch, d_hidden)
        mu_titer = self.mlp_titer(h_pool).squeeze(-1)     # (batch,)
        y_viab   = self.mlp_viab(h_pool).squeeze(-1)      # (batch,)
        return mu_titer, y_viab