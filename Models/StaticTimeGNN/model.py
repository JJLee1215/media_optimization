"""
model.py
StaticTimeGNN — full model assembly

Input:
  m_static   (batch, d_static)          initial media composition
  X_dynamic  (batch, T, d_dynamic)      time series
    d_dynamic = d_dyn_media(9) + d_dyn_feed(4) + d_dyn_process(15) = 28

Output:
  mu_titer  (batch,)    predicted titer
  y_viab    (batch,)    predicted viability
  A_tilde   (batch, N, N)  learned adjacency (for loss + visualization)

Pipeline:
  ① StaticEncoder    m_static → h0, c0
  ② DynamicEncoder   X_dynamic + h0,c0 → H_dynamic (batch, T, d_hidden)
  ③ CrossAttention   H_dynamic + m_static → H_dynamic* (batch, T, d_hidden)
  ④ GraphConstruct   H_dynamic*[-1] → V (batch,N,d_hidden), Ã (batch,N,N)
  ⑤ GNN              V + Ã → H_final (batch, N, d_hidden)
  ⑥ OutputHead       H_final → mu_titer, y_viab
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from .components.encoder   import StaticEncoder, DynamicEncoder
from .components.attention import CrossAttention
from .components.graph     import GraphConstruction, GNN
from .components.output    import OutputHead


class StaticTimeGNN(nn.Module):
    def __init__(self, d_static: int, d_dynamic: int, N: int, A0: torch.Tensor):
        """
        d_static  : number of static features (m_static)
        d_dynamic : number of dynamic features (X_dynamic)
        N         : number of graph nodes (= d_dynamic)
        A0        : (N, N) domain prior adjacency
        """
        super().__init__()
        d_hidden = config.GNN_D_HIDDEN
        n_layers = config.GNN_N_LAYERS
        mlp_hidden = config.GNN_MLP_HIDDEN

        self.static_encoder  = StaticEncoder(d_static, d_hidden)
        self.dynamic_encoder = DynamicEncoder(d_dynamic, d_hidden)
        self.cross_attention = CrossAttention(d_static, d_hidden)
        self.graph_construct = GraphConstruction(d_hidden, N, A0)
        self.gnn             = GNN(d_hidden, n_layers)
        self.output_head     = OutputHead(d_hidden, mlp_hidden)

    def forward(self, m_static: torch.Tensor, X_dynamic: torch.Tensor):
        """
        m_static  : (batch, d_static)
        X_dynamic : (batch, T, d_dynamic)
        Returns:
          mu_titer (batch,)
          y_viab   (batch,)
          A_tilde  (batch, N, N)
        """
        # ① Static encoder
        h0, c0 = self.static_encoder(m_static)

        # ② Dynamic encoder (LSTM initialized with media state)
        H_dynamic = self.dynamic_encoder(X_dynamic, h0, c0)

        # ③ Cross-attention (re-inject media context at every time step)
        H_dynamic_star = self.cross_attention(H_dynamic, m_static)

        # ④ Graph construction
        V, A_tilde = self.graph_construct(H_dynamic_star)

        # ⑤ GNN message passing
        H_final = self.gnn(V, A_tilde)

        # ⑥ Output
        mu_titer, y_viab = self.output_head(H_final)

        return mu_titer, y_viab, A_tilde