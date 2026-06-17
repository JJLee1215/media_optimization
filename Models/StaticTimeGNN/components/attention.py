"""
components/attention.py
Cross-Attention

Re-injects static media information (m_static) into every time step
of H_dynamic to prevent the media context from fading over time.

Input  : H_dynamic (batch, T, d_hidden)  from DynamicEncoder
         m_static  (batch, d_static)     original media composition
Output : H_dynamic* (batch, T, d_hidden) media-context enriched representation

At each time step t:
  Q = H_dynamic[t] · Wq
  K = m_static · Wk
  V = m_static · Wv
  α = softmax(Q · Kᵀ / √d_hidden)
  H_dynamic*[t] = LayerNorm(H_dynamic[t] + α · V)
"""

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, d_static: int, d_hidden: int):
        super().__init__()
        self.Wq    = nn.Linear(d_hidden, d_hidden, bias=False)
        self.Wk    = nn.Linear(d_static,  d_hidden, bias=False)
        self.Wv    = nn.Linear(d_static,  d_hidden, bias=False)
        self.norm  = nn.LayerNorm(d_hidden)
        self.scale = d_hidden ** 0.5

    def forward(self, H_dynamic: torch.Tensor, m_static: torch.Tensor):
        """
        H_dynamic : (batch, T, d_hidden)
        m_static  : (batch, d_static)
        Returns   : (batch, T, d_hidden)
        """
        Q    = self.Wq(H_dynamic)                        # (batch, T, d_hidden)
        K    = self.Wk(m_static).unsqueeze(1)            # (batch, 1, d_hidden)
        V    = self.Wv(m_static).unsqueeze(1)            # (batch, 1, d_hidden)
        attn = torch.softmax(
            torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1
        )                                                 # (batch, T, 1)
        return self.norm(H_dynamic + torch.bmm(attn, V))