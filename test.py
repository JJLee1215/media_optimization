"""
test.py
Model testing entry point

Usage:
  python test.py --model all           test all trained models
  python test.py --model static        GP, RandomForest, XGBoost, MLP
  python test.py --model time          RNN, LSTM, Transformer
  python test.py --model static_time   StaticTimeGNN
  python test.py --model gp            single model
"""

import sys
import os
import argparse
import json
import torch
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


def test_static(model_name):
    save_path = config.model_save_path(model_name)
    if not save_path.exists():
        raise FileNotFoundError(f"No trained model: {save_path}\nRun: python train.py --model {model_name}")

    _, X_test, _, y_test, x_cols, _ = get_static_data()
    model = get_model(model_name)
    model.load()
    return model.evaluate(X_test, y_test)


def test_time(model_name):
    save_path = config.model_save_path(model_name)
    if not save_path.exists():
        raise FileNotFoundError(f"No trained model: {save_path}\nRun: python train.py --model {model_name}")

    _, _, _, X_test, y_test, _ = get_timeseries_data()
    model = get_model(model_name)
    model.load()
    return model.evaluate(X_test, y_test)


def test_static_time():
    from torch.utils.data import Dataset, DataLoader, random_split
    from Models.StaticTimeGNN import StaticTimeGNNModel

    save_path = config.model_save_path("static_time_gnn")
    if not save_path.exists():
        raise FileNotFoundError(f"No trained model: {save_path}\nRun: python train.py --model static_time")

    # Reuse same data loading as train
    import pandas as pd
    import numpy as np

    df_static  = pd.read_csv(config.DATA_STATIC)
    df_dynamic = pd.read_csv(config.DATA_TIMESERIES)

    drop_cols   = ["batch_id", "titer_final", "viab_final"]
    static_cols = [c for c in df_static.columns if c not in drop_cols]

    m_static = torch.tensor(df_static[static_cols].values, dtype=torch.float32)
    y_titer  = torch.tensor(df_static["titer_final"].values, dtype=torch.float32)
    y_viab   = torch.tensor(df_static["viab_final"].values,  dtype=torch.float32)

    batch_col  = "Batch ID"
    time_col   = "Time (day)"
    target_col = "Titer (g/L)"
    skip_cols  = [batch_col, time_col, "Fault flag", target_col]
    feat_cols  = [c for c in df_dynamic.columns if c not in skip_cols]

    X_list = []
    for _, grp in df_dynamic.groupby(batch_col):
        grp = grp.sort_values(time_col)
        X_list.append(grp[feat_cols].values)

    T = min(len(x) for x in X_list)
    X_dynamic = torch.tensor(np.array([x[:T] for x in X_list], dtype=np.float32))

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
    train_loader = DataLoader(train_set, batch_size=config.GNN_BATCH_SIZE)
    val_loader   = DataLoader(val_set,   batch_size=config.GNN_BATCH_SIZE)

    N  = len(feat_cols)
    A0 = torch.zeros(N, N)

    model = StaticTimeGNNModel(
        d_static  = m_static.shape[1],
        d_dynamic = X_dynamic.shape[2],
        N         = N,
        A0        = A0,
    )
    model.load()
    return model.evaluate(train_loader, val_loader)


def test_model(model_name):
    config.make_dirs()
    print(f"\n{'='*55}")
    print(f"  Testing: {model_name.upper()}")
    print(f"{'='*55}")

    try:
        if model_name in STATIC_MODELS:
            result = test_static(model_name)
        elif model_name in TIME_MODELS:
            result = test_time(model_name)
        elif model_name == "static_time_gnn":
            result = test_static_time()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        return None

    # Save result JSON
    result_path = config.result_dir(model_name) / "test_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved: {result_path}")
    return result


def test_group(group: str):
    models = MODEL_GROUPS[group]

    for name in models:
        try:
            test_model(name)
        except Exception as e:
            print(f"\n[{name}] Error: {e}")

    # Compare
    all_results = collect_results("test")
    print_table(all_results, "test")
    plot_comparison(all_results, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Testing")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_GROUPS.keys()) + ALL_MODELS,
        help="Model or group to test"
    )
    args = parser.parse_args()

    if args.model in MODEL_GROUPS:
        test_group(args.model)
    else:
        test_model(args.model)