"""
run.py
StaticTimeGNN quick test entry point

Usage:
  python Models/StaticTimeGNN/run.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

import config
from Models.StaticTimeGNN import StaticTimeGNNModel


# ── Simple Dataset ────────────────────────────

class GNNDataset(torch.utils.data.Dataset):
    def __init__(self, m_static, X_dynamic, y_titer, y_viab):
        self.m  = m_static
        self.X  = X_dynamic
        self.yt = y_titer
        self.yv = y_viab

    def __len__(self): return len(self.m)
    def __getitem__(self, i): return self.m[i], self.X[i], self.yt[i], self.yv[i]


def build_dummy_loaders(d_static, d_dynamic, N=30, T=14):
    torch.manual_seed(42)
    m  = torch.randn(N, d_static)
    X  = torch.randn(N, T, d_dynamic)
    yt = torch.rand(N)
    yv = torch.rand(N) * 0.4 + 0.6

    ds      = GNNDataset(m, X, yt, yv)
    n_train = int(N * 0.8)
    train_set, val_set = random_split(ds, [n_train, N - n_train],
                                      generator=torch.Generator().manual_seed(42))
    return (DataLoader(train_set, batch_size=config.GNN_BATCH_SIZE, shuffle=True),
            DataLoader(val_set,   batch_size=config.GNN_BATCH_SIZE))


if __name__ == "__main__":
    d_static  = 9
    d_dynamic = 28
    N_nodes   = d_dynamic

    # Build A0 (group-based rule)
    A0 = torch.zeros(N_nodes, N_nodes)
    media_idx   = list(range(9))
    feed_idx    = list(range(9, 13))
    process_idx = list(range(13, 28))

    def fill(A, i_list, j_list, val):
        for i in i_list:
            for j in j_list:
                A[i, j] = val; A[j, i] = val

    fill(A0, media_idx,   media_idx,   0.7)
    fill(A0, feed_idx,    media_idx,   0.9)
    fill(A0, feed_idx,    feed_idx,    0.5)
    fill(A0, process_idx, media_idx,   0.4)
    fill(A0, process_idx, process_idx, 0.6)
    fill(A0, process_idx, feed_idx,    0.1)
    for i in range(N_nodes): A0[i, i] = 1.0

    train_loader, val_loader = build_dummy_loaders(d_static, d_dynamic)

    model = StaticTimeGNNModel(d_static, d_dynamic, N_nodes, A0)
    model.train(train_loader, val_loader)
    model.evaluate(train_loader, val_loader)
    model.save()