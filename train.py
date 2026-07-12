"""
train.py
Model training entry point

Usage:
  python train.py --model static
  python train.py --model static --pipeline
  python train.py --model static --static_file my_batch.csv
  python train.py --model time --ts_file my_timeseries.csv
  python train.py --model all --static_file batch.csv --ts_file ts.csv
"""

import sys
import os
import argparse
import json
import datetime
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_preprocess import get_static_data, get_timeseries_data
from compare import collect_results, print_table, plot_comparison

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


def train_static(model_name, use_pipeline=False, static_file=None, selected_cols=None):
    X_train, X_test, y_train, y_test, x_cols, scaler = get_static_data(
        use_pipeline=use_pipeline,
        static_file=static_file,
        selected_cols=selected_cols,
    )
    model = get_model(model_name)
    model.train(X_train, y_train, x_cols=x_cols, scaler=scaler)
    result = model.evaluate(X_test, y_test)
    if hasattr(model, "feature_importance"):
        model.feature_importance()
    if hasattr(model, "cross_validate"):
        model.cross_validate(X_train, y_train)
    model.save(use_pipeline=use_pipeline)

    result["_actual_static_cols"] = x_cols
    result["_model_obj"] = model
    return result


def train_time(model_name, ts_file=None, selected_cols=None):
    loaders, x_sc, y_sc, X_test, y_test, feat_cols = get_timeseries_data(
        ts_file=ts_file,
        selected_cols=selected_cols,
    )
    model = get_model(model_name)
    model.train(loaders, y_sc)
    result = model.evaluate(X_test, y_test)
    model.save()

    result["_actual_ts_cols"] = feat_cols
    result["_model_obj"] = model
    return result


def train_static_time(use_pipeline=False, static_file=None, ts_file=None,
                       selected_cols=None, selected_ts_cols=None):
    """
    StaticTimeGNN — uses both static and timeseries data.

    static_file      : 정적 배지 데이터 파일명 (None이면 기본값)
    ts_file          : 시계열 데이터 파일명 (None이면 기본값)
    selected_cols    : 사용할 static feature 컬럼명 리스트 (None이면 전체 사용)
    selected_ts_cols : 사용할 timeseries feature 컬럼명 리스트 (None이면 전체 사용)

    ※ StaticTimeGNN의 save()는 현재 state_dict만 저장하는 구조라
       use_pipeline 메타정보를 함께 기록하지 못함 (보류 중).
    """
    from torch.utils.data import Dataset, DataLoader, random_split
    from Models.StaticTimeGNN import StaticTimeGNNModel
    import pandas as pd

    # ── 파일 경로 결정 ──
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

    if use_pipeline:
        from heterogeneity.smile_gem_pipe import MediaPipeline
        pipeline    = MediaPipeline()
        m_static_np = pipeline.transform(m_static_np, static_cols)
        print(f"[train_static_time] pipeline applied: {df_static[static_cols].shape} → {m_static_np.shape}")

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

    result["_actual_static_cols"] = static_cols
    result["_actual_ts_cols"]     = feat_cols
    result["_model_obj"]          = model
    return result


def train_model(model_name, use_pipeline=False, static_file=None, ts_file=None,
                 selected_cols=None, selected_ts_cols=None):
    config.make_dirs()
    print(f"\n{'='*55}")
    print(f"  Training    : {model_name.upper()}")
    print(f"  Pipeline    : {'ON' if use_pipeline else 'OFF'}")
    print(f"  Static file : {static_file or config.DATA_STATIC}")
    print(f"  TS file     : {ts_file or config.DATA_TIMESERIES}")
    if selected_cols:
        print(f"  Selected static cols : {selected_cols}")
    if selected_ts_cols:
        print(f"  Selected ts cols     : {selected_ts_cols}")
    print(f"{'='*55}")

    if model_name in STATIC_MODELS:
        result = train_static(model_name, use_pipeline=use_pipeline, static_file=static_file,
                               selected_cols=selected_cols)
    elif model_name in TIME_MODELS:
        result = train_time(model_name, ts_file=ts_file, selected_cols=selected_ts_cols)
    elif model_name == "static_time_gnn":
        result = train_static_time(use_pipeline=use_pipeline, static_file=static_file, ts_file=ts_file,
                                    selected_cols=selected_cols, selected_ts_cols=selected_ts_cols)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # ── 임시로 실어온 내부용 값 꺼내기 ──
    actual_static_cols = result.pop("_actual_static_cols", None)
    actual_ts_cols      = result.pop("_actual_ts_cols", None)
    model_obj           = result.pop("_model_obj", None)

    # ── 학습 정보(meta) 조립 ──
    result["model"]      = model_name
    result["trained_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    result["data_file"]  = {
        "static"    : static_file or config.DATA_STATIC.name,
        "timeseries": ts_file or config.DATA_TIMESERIES.name,
    }
    result["use_pipeline"] = use_pipeline
    result["meta"] = {
        "selected_columns": {
            "static": {
                "requested": selected_cols,
                "actual"   : actual_static_cols,
                "matches"  : (selected_cols is None) or (set(selected_cols or []) == set(actual_static_cols or [])),
            },
            "timeseries": {
                "requested": selected_ts_cols,
                "actual"   : actual_ts_cols,
                "matches"  : (selected_ts_cols is None) or (set(selected_ts_cols or []) == set(actual_ts_cols or [])),
            },
        },
    }
    result["hyperparams"] = model_obj.get_config() if (model_obj and hasattr(model_obj, "get_config")) else {}

    result_path = config.result_dir(model_name) / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved: {result_path}")
    return result


def train_group(group: str, use_pipeline=False, static_file=None, ts_file=None,
                 selected_cols=None, selected_ts_cols=None):
    models  = MODEL_GROUPS[group]
    results = {}

    for name in models:
        try:
            results[name] = train_model(
                name,
                use_pipeline=use_pipeline,
                static_file=static_file,
                ts_file=ts_file,
                selected_cols=selected_cols,
                selected_ts_cols=selected_ts_cols,
            )
        except Exception as e:
            print(f"\n[{name}] Error: {e}")
            results[name] = {"error": str(e)}

    all_results = collect_results("train")
    print_table(all_results, "train")
    plot_comparison(all_results, "train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_GROUPS.keys()) + ALL_MODELS,
    )
    parser.add_argument(
        "--pipeline", action="store_true", default=False,
        help="Enable heterogeneity pipeline (static/static_time only)",
    )
    parser.add_argument(
        "--static_file", type=str, default=None,
        help="Static CSV filename in data_file/ (default: batch_table_syn.csv)",
    )
    parser.add_argument(
        "--ts_file", type=str, default=None,
        help="Timeseries CSV filename in data_file/ (default: timeseries_syn.csv)",
    )
    parser.add_argument(
        "--selected_cols", type=str, default=None,
        help="쉼표로 구분된 static feature 컬럼명 (예: Glucose_0,Glutamine_0)",
    )
    parser.add_argument(
        "--selected_ts_cols", type=str, default=None,
        help="쉼표로 구분된 timeseries feature 컬럼명",
    )
    args = parser.parse_args()

    sel_cols    = args.selected_cols.split(",")    if args.selected_cols    else None
    sel_ts_cols = args.selected_ts_cols.split(",") if args.selected_ts_cols else None

    if args.model in MODEL_GROUPS:
        train_group(
            args.model,
            use_pipeline=args.pipeline,
            static_file=args.static_file,
            ts_file=args.ts_file,
            selected_cols=sel_cols,
            selected_ts_cols=sel_ts_cols,
        )
    else:
        train_model(
            args.model,
            use_pipeline=args.pipeline,
            static_file=args.static_file,
            ts_file=args.ts_file,
            selected_cols=sel_cols,
            selected_ts_cols=sel_ts_cols,
        )