"""
train.py
모델 선택 후 학습 실행

사용법:
  python train.py --model gp
  python train.py --model xgboost
  python train.py --model random_forest
  python train.py --model mlp
  python train.py --model rnn
  python train.py --model lstm
  python train.py --model transformer
  python train.py --model gnn
  python train.py --model all       ← 전체 학습
  python train.py --model all --syn ← 더미 데이터로 전체 학습
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Models/ 폴더를 Python이 찾을 수 있도록 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config

cfg = Config()

STATIC_MODELS     = ["gp", "xgboost", "random_forest", "mlp"]
TIMESERIES_MODELS = ["rnn", "lstm", "transformer"]
GNN_MODELS        = ["gnn"]
ALL_MODELS        = STATIC_MODELS + TIMESERIES_MODELS + GNN_MODELS


def get_model(model_name):
    if model_name == "gp":
        from Models.gp import GPModel
        return GPModel()
    elif model_name == "xgboost":
        from Models.xgboost import XGBoostModel
        return XGBoostModel()
    elif model_name == "random_forest":
        from Models.randomforest import RandomForestModel
        return RandomForestModel()
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
    elif model_name == "gnn":
        from Models.gnn import GNNModel
        return GNNModel()
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}\n지원 모델: {ALL_MODELS}")


def train_model(model_name: str, use_syn: bool = False):
    from data import get_static_data, get_timeseries_data, get_gnn_data

    print(f"\n{'='*55}")
    print(f"  모델: {model_name.upper()}  |  데이터: {'더미' if use_syn else '실제'}")
    print(f"{'='*55}")

    model = get_model(model_name)
    cfg.make_dirs()

    if model_name in STATIC_MODELS:
        X_train, X_test, y_train, y_test, x_cols, scaler = get_static_data(use_syn=use_syn)
        model.train(X_train, y_train, scaler=scaler)
        result = model.evaluate(X_test, y_test)
        if hasattr(model, "cross_validate"):
            model.cross_validate(X_train, y_train)
        if hasattr(model, "feature_importance"):
            model.feature_importance(x_cols)

    elif model_name in TIMESERIES_MODELS:
        loaders, x_scaler, y_scaler, X_test, y_test = get_timeseries_data(use_syn=use_syn)
        model.train(loaders, y_scaler)
        result = model.evaluate(X_test, y_test)

    elif model_name in GNN_MODELS:
        train_loader, val_loader = get_gnn_data(use_syn=use_syn)
        model.train(train_loader, val_loader)
        result = model.evaluate(train_loader, val_loader)

    model.save()

    result_path = cfg.result_dir(model_name) / "train_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n결과 저장: {result_path}")

    return result


def train_all(use_syn: bool = False):
    results = {}

    for model_name in ALL_MODELS:
        try:
            result = train_model(model_name, use_syn=use_syn)
            results[model_name] = result
        except Exception as e:
            print(f"\n[{model_name}] 오류: {e}")
            results[model_name] = {"error": str(e)}

    print(f"\n{'='*55}")
    print("  전체 모델 학습 결과 비교")
    print(f"{'='*55}")
    print(f"  {'모델':<15} {'RMSE':>8}  {'R²':>8}")
    print(f"  {'-'*35}")
    for name, r in results.items():
        if "error" in r:
            print(f"  {name:<15}  오류: {r['error'][:30]}")
        elif "rmse" in r:
            print(f"  {name:<15} {r['rmse']:>8.4f}  {r.get('r2', '-'):>8}")
        elif "titer_rmse" in r:
            print(f"  {name:<15} {r['titer_rmse']:>8.4f}  (titer)")

    all_result_path = cfg.RESULTS_TT_DIR / "all_train_results.json"
    with open(all_result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n전체 결과 저장: {all_result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모델 학습")
    parser.add_argument("--model", type=str, required=True,
                        choices=ALL_MODELS + ["all"])
    parser.add_argument("--syn", action="store_true",
                        help="더미 데이터 사용")
    args = parser.parse_args()

    if args.model == "all":
        train_all(use_syn=args.syn)
    else:
        train_model(args.model, use_syn=args.syn)