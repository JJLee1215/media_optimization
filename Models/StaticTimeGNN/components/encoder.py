"""
components/encoder.py
Static and Dynamic encoders

StaticEncoder:
  Input  : m_static  (batch, d_static)
  Output : h0        (1, batch, d_hidden)
           c0        (1, batch, d_hidden)
  Role   : converts initial media composition into LSTM initial state

DynamicEncoder:
  Input  : X_dynamic (batch, T, d_dynamic)
           h0, c0    (1, batch, d_hidden)  from StaticEncoder
  Output : H_dynamic (batch, T, d_hidden)
  Role   : reads time series conditioned on initial media state
"""

import torch
import torch.nn as nn


class StaticEncoder(nn.Module):
    def __init__(self, d_static: int, d_hidden: int):
        super().__init__()
        self.h0 = nn.Sequential(nn.Linear(d_static, d_hidden), nn.Tanh())
        self.c0 = nn.Sequential(nn.Linear(d_static, d_hidden), nn.Tanh())

    def forward(self, m_static: torch.Tensor):
        """
        m_static : (batch, d_static)
        Returns  : h0 (1, batch, d_hidden)
                   c0 (1, batch, d_hidden)
        """
        return self.h0(m_static).unsqueeze(0), self.c0(m_static).unsqueeze(0)


class DynamicEncoder(nn.Module):
    def __init__(self, d_dynamic: int, d_hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(d_dynamic, d_hidden, batch_first=True)

    def forward(self, X_dynamic: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor):
        """
        X_dynamic : (batch, T, d_dynamic)
        h0, c0    : (1, batch, d_hidden)
        Returns   : H_dynamic (batch, T, d_hidden)
        """
        H, _ = self.lstm(X_dynamic, (h0, c0))
        return H