"""
StaticTimeGNN
  Static media composition (m_static) + Time series process data (X_dynamic)
  → Graph Neural Network → titer + viability prediction

Usage:
  from Models.StaticTimeGNN import StaticTimeGNNModel
  model = StaticTimeGNNModel()
  model.train(train_loader, val_loader)
"""

from .train_module import StaticTimeGNNModel

__all__ = ["StaticTimeGNNModel"]