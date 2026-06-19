"""
data_preprocess.py
Data loading and preprocessing

Add a new function here when adding a new model.

Functions:
  get_static_data()      for GP, XGBoost, RandomForest, MLP
  get_timeseries_data()  for RNN, LSTM, Transformer
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

import config


def get_static_data():
    """
    Load and preprocess static data for GP, XGBoost, RandomForest, MLP.

    Source : config.DATA_STATIC (batch_table_syn.csv)
    Input  : initial media composition (9 features)
    Target : titer_final

    Returns:
        X_train  (n_train, 9)
        X_test   (n_test,  9)
        y_train  (n_train,)
        y_test   (n_test,)
        x_cols   list of feature names
        scaler   fitted StandardScaler
    """
    df = pd.read_csv(config.DATA_STATIC)

    drop_cols = ["Batch_ID", "titer_final", "viab_final"]
    x_cols = [c for c in df.columns if c not in drop_cols]

    X = df[x_cols].values.astype(np.float32)
    y = df["titer_final"].values.astype(np.float32)

    print(f"[preprocess] Loaded: {config.DATA_STATIC}")
    print(f"  X: {X.shape}  y: {y.shape}")
    print(f"  Features: {x_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"  train={len(X_train)}  test={len(X_test)}")
    return X_train, X_test, y_train, y_test, x_cols, scaler


def get_timeseries_data(batch_size=None):
    """
    Load and preprocess time series data for RNN, LSTM, Transformer.

    Source : config.DATA_TIMESERIES (timeseries_syn.csv)

    Input  : X (n_batches, T, d_dynamic)
               d_dynamic = d_dyn_media + d_dyn_feed + d_dyn_process
               T = sequence length (days)

    Target : y (n_batches,)   titer at last time step

    Returns:
        loaders      {"train": DataLoader, "val": DataLoader}
        x_scaler     fitted StandardScaler for X
        y_scaler     fitted StandardScaler for y
        X_test       (n_test, T, d_dynamic)
        y_test       (n_test,)
        feat_cols    list of feature names
    """
    if batch_size is None:
        batch_size = config.RNN_BATCH_SIZE

    df = pd.read_csv(config.DATA_TIMESERIES)

    # Remove fault batches
    if "Fault flag" in df.columns:
        df = df[df["Fault flag"] == 0]

    batch_col  = "Batch_ID"
    time_col   = "Time (day)"
    target_col = "Titer (g/L)"

    # Feature columns (exclude meta columns)
    skip_cols  = [batch_col, time_col, "Fault flag", target_col]
    feat_cols  = [c for c in df.columns if c not in skip_cols]

    # Build sequences
    X_list, y_list = [], []
    for _, grp in df.groupby(batch_col):
        grp = grp.sort_values(time_col)
        X_list.append(grp[feat_cols].values.astype(np.float32))
        # titer at last time step
        titer_vals = grp[target_col].values
        y_list.append(float(titer_vals[titer_vals > 0][-1]
                            if any(titer_vals > 0) else titer_vals[-1]))

    T   = min(len(x) for x in X_list)   # use shortest sequence length
    X   = np.array([x[:T] for x in X_list], dtype=np.float32)
    y   = np.array(y_list, dtype=np.float32)

    print(f"[preprocess] Loaded: {config.DATA_TIMESERIES}")
    print(f"  X: {X.shape}  (n_batches, T, d_dynamic)")
    print(f"  y: {y.shape}")
    print(f"  d_dynamic breakdown:")
    print(f"    feat_cols ({len(feat_cols)}): {feat_cols}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=config.VAL_SIZE, random_state=config.RANDOM_SEED
    )
    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    # Scale X: fit on train, apply to all
    n_feat   = X.shape[2]
    x_scaler = StandardScaler()
    x_scaler.fit(X_train.reshape(-1, n_feat))
    X_train  = x_scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_val    = x_scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
    X_test   = x_scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

    # Scale y
    y_scaler  = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s   = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

    def to_loader(X, y, shuffle=False):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    loaders = {
        "train": to_loader(X_train, y_train_s, shuffle=True),
        "val"  : to_loader(X_val,   y_val_s),
    }

    return loaders, x_scaler, y_scaler, X_test, y_test, feat_cols


if __name__ == "__main__":
    print("=" * 50)
    print("[1] Static data")
    X_train, X_test, y_train, y_test, x_cols, scaler = get_static_data()
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

    print("\n[2] Time series data")
    loaders, x_sc, y_sc, X_test_ts, y_test_ts, feat_cols = get_timeseries_data()
    xb, yb = next(iter(loaders["train"]))
    print(f"  batch X: {xb.shape}  batch y: {yb.shape}")