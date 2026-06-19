"""
train.py
Model training entry point

Usage:
  python train.py --model all           train all models
  python train.py --model static        GP, RandomForest, XGBoost, MLP
  python train.py --model time          RNN, LSTM, Transformer
  python train.py --model static_time   StaticTimeGNN
  python train.py --model gp            single model
"""

import sys
import os
import argparse
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_preprocess import get_static_data, get_timeseries_data
from compare import collect_results, print_table, plot_comparison

STATIC_MODELS      = ["gaussian_process", "random_forest", "xgboost", "mlp"]
TIME_MODELS        = ["rnn", "lstm", "transformer"]
STATIC_TIME_MODELS = ["static_time_gnn"]
ALL_MODELS         = STATIC_MODELS + TIME_MODELS + STATIC_TIME_MODELS

MODEL_GROUPS = {
    "static"      : STATIC_MODELS,
    "time"        : TIME_MODELS,
    "static_time" : STATIC_TIME_MODELS,
    "all"         : ALL_MODELS,
}


def get_model(model_name):
    if model_name == "gaussian_process":
        from Models.gaussian_process import GaussianProcessModel
        return GaussianProcessModel()
    elif model_name == "random_forest":
        from Models.randomforest import RandomForestModel
        return RandomForestModel()
    elif model_name == "xgboost":
        from Models.xgboost_model import XGBoostModel
        return XGBoostModel()
    elif model_name == "mlp":
        from Models.mlp import MLPModel
        return MLPModel()
    elif model_name == "rnn":
        from Models.rnn import RNNModel
        return RNNModel()
    elif model_name == "lstm":
        from Models.lstm import LSTMModel
        return LSTMModel()
    elif model_name == "transformer":
        from Models.transformer import TransformerModel
        return TransformerModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_static(model_name):
    X_train, X_test, y_train, y_test, x_cols, scaler = get_static_data()
    model = get_model(model_name)
    model.train(X_train, y_train, x_cols=x_cols, scaler=scaler)
    result = model.evaluate(X_test, y_test)
    if hasattr(model, "feature_importance"):
        model.feature_importance()
    if hasattr(model, "cross_validate"):
        model.cross_validate(X_train, y_train)
    model.save()
    return result


def train_time(model_name):
    loaders, x_sc, y_sc, X_test, y_test, feat_cols = get_timeseries_data()
    model = get_model(model_name)
    model.train(loaders, y_sc)
    result = model.evaluate(X_test, y_test)
    model.save()
    return result


def train_static_time():
    """StaticTimeGNN — uses both static and timeseries data."""
    from torch.utils.data import Dataset, DataLoader, random_split
    from Models.StaticTimeGNN import StaticTimeGNNModel

    # Load data
    df_static  = __import__("pandas").read_csv(config.DATA_STATIC)
    df_dynamic = __import__("pandas").read_csv(config.DATA_TIMESERIES)

    drop_cols  = ["Batch_ID", "titer_final", "viab_final"]
    static_cols = [c for c in df_static.columns if c not in drop_cols]

    m_static  = torch.tensor(df_static[static_cols].values, dtype=torch.float32)
    y_titer   = torch.tensor(df_static["titer_final"].values, dtype=torch.float32)
    y_viab    = torch.tensor(df_static["viab_final"].values,  dtype=torch.float32)

    # Time series
    batch_col  = "Batch_ID"
    time_col   = "Time (day)"
    target_col = "Titer (g/L)"
    skip_cols  = [batch_col, time_col, "Fault flag", target_col]
    feat_cols  = [c for c in df_dynamic.columns if c not in skip_cols]

    X_list = []
    for _, grp in df_dynamic.groupby(batch_col):
        grp = grp.sort_values(time_col)
        X_list.append(grp[feat_cols].values)

    T = min(len(x) for x in X_list)
    X_dynamic = torch.tensor(
        __import__("numpy").array([x[:T] for x in X_list], dtype=__import__("numpy").float32)
    )

    # Dataset
    class GNNDataset(Dataset):
        def __init__(self, m, X, yt, yv):
            self.m, self.X, self.yt, self.yv = m, X, yt, yv
        def __len__(self): return len(self.m)
        def __getitem__(self, i): return self.m[i], self.X[i], self.yt[i], self.yv[i]

    dataset = GNNDataset(m_static, X_dynamic, y_titer, y_viab)
    n_train = int(len(dataset) * config.TRAIN_RATIO)
    n_val   = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    train_loader = DataLoader(train_set, batch_size=config.GNN_BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=config.GNN_BATCH_SIZE)

    # Build A0
    N     = len(feat_cols)
    A0    = torch.zeros(N, N)
    media_idx   = list(range(9))
    feed_idx    = list(range(9, 13))
    process_idx = list(range(13, N))

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
    for i in range(N): A0[i, i] = 1.0

    # Train
    model = StaticTimeGNNModel(
        d_static  = m_static.shape[1],
        d_dynamic = X_dynamic.shape[2],
        N         = N,
        A0        = A0,
    )
    model.train(train_loader, val_loader)
    result = model.evaluate(train_loader, val_loader)
    model.save()
    return result


def train_model(model_name):
    config.make_dirs()
    print(f"\n{'='*55}")
    print(f"  Training: {model_name.upper()}")
    print(f"{'='*55}")

    if model_name in STATIC_MODELS:
        result = train_static(model_name)
    elif model_name in TIME_MODELS:
        result = train_time(model_name)
    elif model_name == "static_time_gnn":
        result = train_static_time()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Save result JSON
    result_path = config.result_dir(model_name) / "train_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved: {result_path}")
    return result


def train_group(group: str):
    models  = MODEL_GROUPS[group]
    results = {}

    for name in models:
        try:
            results[name] = train_model(name)
        except Exception as e:
            print(f"\n[{name}] Error: {e}")
            results[name] = {"error": str(e)}

    # Compare
    all_results = collect_results("train")
    print_table(all_results, "train")
    plot_comparison(all_results, "train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_GROUPS.keys()) + ALL_MODELS,
        help="Model or group to train"
    )
    args = parser.parse_args()

    if args.model in MODEL_GROUPS:
        train_group(args.model)
    else:
        train_model(args.model)