"""
components/graph.py
Graph Construction + GNN Reasoning

GraphConstruction:
  Learns adjacency matrix from data, constrained by causal mask.
  A0 is used only in Loss (L_graph regularization).

  Formula:
    A_raw   = ( vᵢWₐ · vⱼWₐᵀ ) / √d_hidden
    A_tilde = sigmoid( A_raw - bias ) ⊙ M_causal
              ↑ bias shifts sigmoid center → values spread across [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConstruction(nn.Module):
    def __init__(self, d_hidden: int, N: int, A0: torch.Tensor):
        super().__init__()
        self.scale = d_hidden ** 0.5

        # Wa with small init → small dot products → sigmoid near 0.5
        self.Wa = nn.Linear(d_hidden, d_hidden, bias=False)
        nn.init.xavier_uniform_(self.Wa.weight, gain=0.1)

        # Learnable bias to shift sigmoid center
        self.bias = nn.Parameter(torch.zeros(1))

        self.register_buffer("M_causal", torch.ones(N, N))
        self.register_buffer("A0", A0)

    def forward(self, H_dynamic_star: torch.Tensor):
        """
        H_dynamic_star : (batch, T, d_hidden)
        Returns:
          V       (batch, N, d_hidden)
          A_tilde (batch, N, N)   values spread across [0, 1]
        """
        N = self.A0.shape[0]
        V = H_dynamic_star[:, -1, :].unsqueeze(1).expand(-1, N, -1)

        V_proj  = self.Wa(V)
        A_raw   = torch.bmm(V_proj, V_proj.transpose(1, 2)) / self.scale
        A       = torch.sigmoid(A_raw - self.bias)
        A_tilde = A * self.M_causal

        return V, A_tilde


class GNNLayer(nn.Module):
    def __init__(self, d_hidden: int):
        super().__init__()
        self.W = nn.Linear(d_hidden, d_hidden)

    def forward(self, H: torch.Tensor, A_tilde: torch.Tensor):
        deg    = A_tilde.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        A_norm = A_tilde / deg
        return F.relu(self.W(torch.bmm(A_norm, H)))


class GNN(nn.Module):
    def __init__(self, d_hidden: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([GNNLayer(d_hidden) for _ in range(n_layers)])

    def forward(self, V: torch.Tensor, A_tilde: torch.Tensor):
        H = V
        for layer in self.layers:
            H = layer(H, A_tilde)
        return H