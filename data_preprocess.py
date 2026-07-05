"""
data_preprocess.py
Data loading and preprocessing

Functions:
  get_static_data()      GP, XGBoost, RandomForest, MLP
  get_timeseries_data()  RNN, LSTM, Transformer

※ 파일 경로 처리 방식:
  static_file / ts_file 파라미터로 파일명을 받아
  config.DATA_DIR / {파일명} 경로로 읽음.
  파라미터가 없으면 config.DATA_STATIC / config.DATA_TIMESERIES 기본값 사용.
  → UI에서 업로드한 파일명을 train.py → data_preprocess.py로 전달하면
    하드코딩 없이 어떤 CSV든 사용 가능.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from heterogeneity.smile_gem_pipe import MediaPipeline


# 파이프라인 싱글톤 (모듈 로드 시 1회만 초기화)
_pipeline = None

def get_pipeline() -> MediaPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MediaPipeline()
    return _pipeline


def get_static_data(use_pipeline: bool = False, static_file: str = None):
    """
    Load and preprocess static data for GP, XGBoost, RandomForest, MLP.

    static_file  : 파일명 (예: "batch_table_syn.csv")
                   None이면 config.DATA_STATIC 기본값 사용
                   UI에서 업로드한 파일명을 전달하면 해당 파일로 학습

    use_pipeline : True → 9 → 230차원 heterogeneity pipeline 적용
                   False → 9 features 그대로 (기본값)

    Returns:
        X_train  (n_train, VECTOR_DIM) or (n_train, 9)
        X_test   (n_test,  VECTOR_DIM) or (n_test,  9)
        y_train  (n_train,)
        y_test   (n_test,)
        x_cols   list of feature names
        scaler   fitted StandardScaler
    """
    # ── 파일 경로 결정 ──
    if static_file:
        file_path = config.DATA_DIR / static_file
    else:
        file_path = config.DATA_STATIC   # 기본값: data_file/batch_table_syn.csv

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Static file not found: {file_path}")

    df = pd.read_csv(file_path)

    drop_cols = ["Batch_ID", "titer_final", "viab_final"]
    x_cols    = [c for c in df.columns if c not in drop_cols]

    X = df[x_cols].values.astype(np.float32)
    y = df["titer_final"].values.astype(np.float32)

    print(f"[preprocess] Loaded: {file_path}")
    print(f"  Raw X: {X.shape}  y: {y.shape}")
    print(f"  Features: {x_cols}")

    # ── Step 1~4: media pipeline ──────────────
    if use_pipeline:
        pipeline = get_pipeline()
        X = pipeline.transform(X, x_cols)
        print(f"  Pipeline X: {X.shape}")

    # ── train / test split ────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )

    # ── scale ─────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"  train={len(X_train)}  test={len(X_test)}")
    return X_train, X_test, y_train, y_test, x_cols, scaler


def get_timeseries_data(batch_size=None, ts_file: str = None):
    """
    Load and preprocess time series data for RNN, LSTM, Transformer.

    ts_file  : 파일명 (예: "timeseries_syn.csv")
               None이면 config.DATA_TIMESERIES 기본값 사용
               UI에서 업로드한 파일명을 전달하면 해당 파일로 학습

    ※ 시계열 pipeline 적용은 추후 고려, 현재는 기존 동작 유지

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

    # ── 파일 경로 결정 ──
    if ts_file:
        file_path = config.DATA_DIR / ts_file
    else:
        file_path = config.DATA_TIMESERIES   # 기본값: data_file/timeseries_syn.csv

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Timeseries file not found: {file_path}")

    df = pd.read_csv(file_path)

    if "Fault flag" in df.columns:
        df = df[df["Fault flag"] == 0]

    batch_col  = "Batch_ID"
    time_col   = "Time (day)"
    target_col = "Titer (g/L)"
    skip_cols  = [batch_col, time_col, "Fault flag", target_col]
    feat_cols  = [c for c in df.columns if c not in skip_cols]

    X_list, y_list = [], []
    for _, grp in df.groupby(batch_col):
        grp = grp.sort_values(time_col)
        X_list.append(grp[feat_cols].values.astype(np.float32))
        titer_vals = grp[target_col].values
        y_list.append(float(
            titer_vals[titer_vals > 0][-1]
            if any(titer_vals > 0) else titer_vals[-1]
        ))

    T = min(len(x) for x in X_list)
    X = np.array([x[:T] for x in X_list], dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"[preprocess] Loaded: {file_path}")
    print(f"  X: {X.shape}  (n_batches, T, d_dynamic)")
    print(f"  y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=config.VAL_SIZE, random_state=config.RANDOM_SEED
    )
    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    n_feat   = X.shape[2]
    x_scaler = StandardScaler()
    x_scaler.fit(X_train.reshape(-1, n_feat))
    X_train  = x_scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_val    = x_scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
    X_test   = x_scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

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
    print("=" * 55)
    print("[1] Static data (pipeline=False, 기본값)")
    X_train, X_test, y_train, y_test, x_cols, scaler = get_static_data()
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

    print("\n[2] Static data (pipeline=True)")
    X_train2, X_test2, _, _, _, _ = get_static_data(use_pipeline=True)
    print(f"  X_train: {X_train2.shape}  X_test: {X_test2.shape}")

    print("\n[3] Timeseries data")
    loaders, x_sc, y_sc, X_test_ts, y_test_ts, feat_cols = get_timeseries_data()
    xb, yb = next(iter(loaders["train"]))
    print(f"  batch X: {xb.shape}  batch y: {yb.shape}")