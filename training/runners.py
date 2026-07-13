"""
training/runners.py
모델 타입별 실제 학습 실행 로직 — get_model(), train_static(), train_time(), train_static_time()

train.py에서 이 함수들을 호출해서 학습을 실행함. "무엇을 어떻게 학습시키는가"만
담당하고, 결과를 어떤 JSON 형태로 보고할지는 training/result_schema.py가 담당.

※ 각 train_* 함수는 학습 결과 dict에 아래 내부용 키를 임시로 실어서 반환함
  (train.py의 train_model()이 .pop()으로 꺼내서 result_schema 조립에 사용):
    _actual_static_cols, _actual_ts_cols
    _actual_embedding_model, _actual_pipeline_dim
    _model_obj
"""

import torch
import numpy as np
from pathlib import Path

import config
from data_preprocess import get_static_data, get_timeseries_data

STATIC_MODELS      = ["gaussian_process", "random_forest", "xgboost", "mlp"]
TIME_MODELS        = ["rnn", "lstm", "transformer", "tcn"]
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
    elif model_name == "tcn":
        from Models.tcn import TCNModel
        return TCNModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_static(model_name, use_pipeline=False, static_file=None, selected_cols=None,
                  embedding_model=None):
    """
    embedding_model : "rdkit" | "chemberta" | "unimol" | None
                      get_static_data()에 그대로 전달됨.
    """
    X_train, X_test, y_train, y_test, x_cols, scaler, actual_embedding_model, pipeline_dim = \
        get_static_data(
            use_pipeline=use_pipeline,
            static_file=static_file,
            selected_cols=selected_cols,
            embedding_model=embedding_model,
        )
    model = get_model(model_name)
    model.train(X_train, y_train, x_cols=x_cols, scaler=scaler)
    result = model.evaluate(X_test, y_test)
    if hasattr(model, "feature_importance"):
        model.feature_importance()
    if hasattr(model, "cross_validate"):
        model.cross_validate(X_train, y_train)
    model.save(use_pipeline=use_pipeline)

    result["_actual_static_cols"]      = x_cols
    result["_actual_embedding_model"]  = actual_embedding_model
    result["_actual_pipeline_dim"]     = pipeline_dim
    result["_model_obj"]               = model
    return result


def train_time(model_name, ts_file=None, selected_cols=None):
    """※ timeseries는 아직 pipeline 미지원 (embedding_model 파라미터 없음)."""
    loaders, x_sc, y_sc, X_test, y_test, feat_cols = get_timeseries_data(
        ts_file=ts_file, selected_cols=selected_cols,
    )
    model = get_model(model_name)
    model.train(loaders, y_sc)
    result = model.evaluate(X_test, y_test)
    model.save()

    result["_actual_ts_cols"] = feat_cols
    result["_model_obj"]      = model
    return result


def train_static_time(use_pipeline=False, static_file=None, ts_file=None,
                       selected_cols=None, selected_ts_cols=None, embedding_model=None):
    """
    StaticTimeGNN — uses both static and timeseries data.
    ※ timeseries feature selection은 인접행렬(A0) 인덱스 하드코딩 때문에
      아직 안전하게 지원되지 않음 (별도 이슈, 미해결).
    """
    from torch.utils.data import Dataset, DataLoader, random_split
    from Models.StaticTimeGNN import StaticTimeGNNModel
    from heterogeneity._registry import get_pipeline, get_pipeline_dim_info
    import pandas as pd

    static_path = config.DATA_DIR / static_file if static_file else config.DATA_STATIC
    ts_path     = config.DATA_DIR / ts_file     if ts_file     else config.DATA_TIMESERIES

    if not Path(static_path).exists():
        raise FileNotFoundError(f"Static file not found: {static_path}")
    if not Path(ts_path).exists():
        raise FileNotFoundError(f"Timeseries file not found: {ts_path}")

    df_static  = pd.read_csv(static_path)
    df_dynamic = pd.read_csv(ts_path)

    drop_cols   = ["Batch_ID", "titer_final", "viab_final"]
    static_cols = [c for c in df_static.columns if c not in drop_cols]
    if selected_cols:
        static_cols = [c for c in static_cols if c in selected_cols]

    m_static_np = df_static[static_cols].values.astype(np.float32)

    actual_embedding_model = None
    pipeline_dim            = None
    if use_pipeline:
        pipeline = get_pipeline(embedding_model)
        actual_embedding_model = embedding_model or "rdkit"
        m_static_np = pipeline.transform(m_static_np, static_cols)
        pipeline_dim = get_pipeline_dim_info(actual_embedding_model, pipeline)
        print(f"[train_static_time] pipeline applied: ({len(static_cols)},) → {m_static_np.shape}  (embedding_model={actual_embedding_model})")

    m_static = torch.tensor(m_static_np, dtype=torch.float32)
    y_titer  = torch.tensor(df_static["titer_final"].values, dtype=torch.float32)
    y_viab   = torch.tensor(df_static["viab_final"].values,  dtype=torch.float32)

    batch_col  = "Batch_ID"
    time_col   = "Time (day)"
    target_col = "Titer (g/L)"
    skip_cols  = [batch_col, time_col, "Fault flag", target_col]
    feat_cols  = [c for c in df_dynamic.columns if c not in skip_cols]
    if selected_ts_cols:
        feat_cols = [c for c in feat_cols if c in selected_ts_cols]

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
    train_loader = DataLoader(train_set, batch_size=config.GNN_BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=config.GNN_BATCH_SIZE)

    N  = len(feat_cols)
    A0 = torch.zeros(N, N)
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

    model = StaticTimeGNNModel(
        d_static  = m_static.shape[1],
        d_dynamic = X_dynamic.shape[2],
        N         = N,
        A0        = A0,
    )
    model.train(train_loader, val_loader)
    result = model.evaluate(train_loader, val_loader)
    model.save()

    result["_actual_static_cols"]     = static_cols
    result["_actual_ts_cols"]         = feat_cols
    result["_actual_embedding_model"] = actual_embedding_model
    result["_actual_pipeline_dim"]    = pipeline_dim
    result["_model_obj"]              = model
    return result