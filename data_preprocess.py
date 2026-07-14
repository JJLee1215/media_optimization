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

※ selected_cols 파라미터:
  UI Feature Selection에서 고른 컬럼명 리스트. None이면 전체 컬럼 사용.

※ embedding_model / pooling_method / use_pca / other_blocks 파라미터:
  UI Heterogeneity 카드에서 고른 값들. get_pipeline()에 그대로 전달됨.
  other_blocks : ["log_conc", "metal_physchem", "gem"] 중 concat에 포함할 것들.
                 None이면 셋 다 포함(기본값).

※ Split 순서 (중요, 어제 설계 확정):
  raw 농도값 기준으로 먼저 train/test를 나누고, 그 다음에 각각
  파이프라인(notation→embedding→concat→pooling→scaler→PCA)을 통과시킴.
    - train : pipeline.fit_transform()  → scaler/PCA가 여기서 학습(fit)됨
    - test  : pipeline.transform()      → 학습된 규칙을 적용만, 절대 fit 안 함
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from heterogeneity._registry import get_pipeline


def get_static_data(use_pipeline: bool = False, static_file: str = None,
                     selected_cols: list = None, embedding_model: str = None,
                     pooling_method: str = "mean", use_pca: bool = False, pca_dim: int = 30,
                     other_blocks: list = None):
    """
    Load and preprocess static data for GP, XGBoost, RandomForest, MLP.

    use_pipeline    : True → heterogeneity pipeline 적용
    embedding_model : "rdkit" | "chemberta" | "unimol" | None(=rdkit)
    pooling_method  : "mean" | "multi_stat"
    use_pca         : PCA로 최종 차원 압축 여부
    pca_dim         : use_pca=True일 때 목표 차원
    other_blocks    : ["log_conc", "metal_physchem", "gem"] 중 concat에 포함할 것들.
                       None이면 셋 다 포함.

    Returns:
        X_train, X_test, y_train, y_test, x_cols, scaler,
        actual_embedding_model  (use_pipeline=False면 None)
        pipeline_dim             (use_pipeline=False면 None)
    """
    if static_file:
        file_path = config.DATA_DIR / static_file
    else:
        file_path = config.DATA_STATIC

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Static file not found: {file_path}")

    df = pd.read_csv(file_path)

    drop_cols = ["Batch_ID", "titer_final", "viab_final"]
    x_cols    = [c for c in df.columns if c not in drop_cols]

    if selected_cols:
        x_cols = [c for c in x_cols if c in selected_cols]
        if not x_cols:
            raise ValueError("selected_cols와 일치하는 static feature가 없습니다.")

    X = df[x_cols].values.astype(np.float32)
    y = df["titer_final"].values.astype(np.float32)

    print(f"[preprocess] Loaded: {file_path}")
    print(f"  Raw X: {X.shape}  y: {y.shape}")
    print(f"  Features: {x_cols}")

    # ── Step 0: split을 파이프라인보다 먼저 수행 ──────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED,
    )

    actual_embedding_model = None
    pipeline_dim            = None

    if use_pipeline:
        pipeline = get_pipeline(embedding_model, pooling_method=pooling_method,
                                  use_pca=use_pca, pca_dim=pca_dim, other_blocks=other_blocks)
        actual_embedding_model = embedding_model or "rdkit"

        # ── Step 1~5: notation → embedding → concat → pooling → scaler(→PCA) ──
        X_train = pipeline.fit_transform(X_train_raw, x_cols)
        X_test  = pipeline.transform(X_test_raw, x_cols)

        pipeline_dim = {
            "embedding"      : pipeline._emb_dim,
            "metal_physchem" : 5 if "metal_physchem" in pipeline.other_blocks else 0,
            "log_conc"       : 1 if "log_conc" in pipeline.other_blocks else 0,
            "gem"            : 7 if "gem" in pipeline.other_blocks else 0,
            "pooling_method" : pipeline.pooling_method,
            "other_blocks"   : pipeline.other_blocks,
            "pooled_dim"     : pipeline.pooled_dim,
            "use_pca"        : pipeline.use_pca,
            "total"          : pipeline.vector_dim,
        }
        scaler = pipeline.scaler

        print(f"  Pipeline X: train {X_train.shape}, test {X_test.shape}"
              f"  (embedding_model={actual_embedding_model}, pooling={pooling_method}, "
              f"pca={use_pca}, other_blocks={pipeline.other_blocks})")

    else:
        # ── pipeline 미사용: raw feature에 scaler만 적용 (기존 동작 그대로) ──
        scaler   = StandardScaler()
        X_train  = scaler.fit_transform(X_train_raw)
        X_test   = scaler.transform(X_test_raw)

    print(f"  train={len(X_train)}  test={len(X_test)}")
    return X_train, X_test, y_train, y_test, x_cols, scaler, actual_embedding_model, pipeline_dim


def get_timeseries_data(batch_size=None, ts_file: str = None, selected_cols: list = None):
    """
    Load and preprocess time series data for RNN, LSTM, Transformer.
    ※ 시계열 pipeline은 아직 미지원 (embedding_model 파라미터 없음).

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

    if ts_file:
        file_path = config.DATA_DIR / ts_file
    else:
        file_path = config.DATA_TIMESERIES

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

    if selected_cols:
        feat_cols = [c for c in feat_cols if c in selected_cols]
        if not feat_cols:
            raise ValueError("selected_cols와 일치하는 timeseries feature가 없습니다.")

    X_list, y_list = [], []
    for _, grp in df.groupby(batch_col):
        grp = grp.sort_values(time_col)
        X_list.append(grp[feat_cols].values.astype(np.float32))
        titer_vals = grp[target_col].values
        y_list.append(float(
            titer_vals[titer_vals > 0][-1] if any(titer_vals > 0) else titer_vals[-1]
        ))

    T = min(len(x) for x in X_list)
    X = np.array([x[:T] for x in X_list], dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"[preprocess] Loaded: {file_path}")
    print(f"  X: {X.shape}  (n_batches, T, d_dynamic)")
    print(f"  Features: {feat_cols}")
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
    print("[1] Static data (pipeline=False)")
    r = get_static_data()
    print(f"  X_train: {r[0].shape}  embedding_model: {r[6]}  dim: {r[7]}")

    print("\n[2] Static data (pipeline=True, rdkit, mean, other_blocks=gem)")
    r = get_static_data(use_pipeline=True, embedding_model="rdkit",
                          pooling_method="mean", other_blocks=["gem"])
    print(f"  X_train: {r[0].shape}  X_test: {r[1].shape}  embedding_model: {r[6]}")
    print(f"  dim: {r[7]}")

    print("\n[3] Static data (pipeline=True, rdkit, multi_stat, PCA on)")
    r = get_static_data(use_pipeline=True, embedding_model="rdkit",
                          pooling_method="multi_stat", use_pca=True, pca_dim=30)
    print(f"  X_train: {r[0].shape}  X_test: {r[1].shape}  embedding_model: {r[6]}")
    print(f"  dim: {r[7]}")

    print("\n[4] Timeseries data")
    loaders, x_sc, y_sc, X_test_ts, y_test_ts, feat_cols = get_timeseries_data()
    xb, yb = next(iter(loaders["train"]))
    print(f"  batch X: {xb.shape}  batch y: {yb.shape}")