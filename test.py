"""
test.py
학습된 모델 로드 후 테스트 실행

사용법:
  python test.py --model gp
  python test.py --model gnn
  python test.py --model all       ← 전체 테스트 + 비교
  python test.py --model all --syn ← 더미 데이터로 전체 테스트
"""

import sys
import os
import argparse
import json
from pathlib import Path

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
        raise ValueError(f"지원하지 않는 모델: {model_name}")


def test_model(model_name: str, use_syn: bool = False):
    from data import get_static_data, get_timeseries_data, get_gnn_data

    save_path = cfg.model_save_path(model_name)
    if not save_path.exists():
        raise FileNotFoundError(
            f"학습된 모델 없음: {save_path}\n"
            f"먼저 python train.py --model {model_name} 실행"
        )

    print(f"\n{'='*55}")
    print(f"  테스트: {model_name.upper()}  |  데이터: {'더미' if use_syn else '실제'}")
    print(f"{'='*55}")

    model = get_model(model_name)
    model.load()

    if model_name in STATIC_MODELS:
        X_train, X_test, y_train, y_test, x_cols, scaler = get_static_data(use_syn=use_syn)
        result = model.evaluate(X_test, y_test)

    elif model_name in TIMESERIES_MODELS:
        loaders, x_scaler, y_scaler, X_test, y_test = get_timeseries_data(use_syn=use_syn)
        result = model.evaluate(X_test, y_test)

    elif model_name in GNN_MODELS:
        train_loader, val_loader = get_gnn_data(use_syn=use_syn)
        result = model.evaluate(train_loader, val_loader)

    result_path = cfg.result_dir(model_name) / "test_result.json"
    cfg.result_dir(model_name).mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"결과 저장: {result_path}")

    return result


def test_all(use_syn: bool = False):
    results = {}

    for model_name in ALL_MODELS:
        try:
            result = test_model(model_name, use_syn=use_syn)
            results[model_name] = result
        except FileNotFoundError as e:
            print(f"\n[{model_name}] 스킵: 모델 파일 없음")
            results[model_name] = {"skipped": "모델 파일 없음"}
        except Exception as e:
            print(f"\n[{model_name}] 오류: {e}")
            results[model_name] = {"error": str(e)}

    print(f"\n{'='*55}")
    print("  전체 모델 테스트 결과 비교")
    print(f"{'='*55}")
    print(f"  {'모델':<15} {'RMSE':>10}  {'R²':>8}")
    print(f"  {'-'*38}")

    ranked = []
    for name, r in results.items():
        if "rmse" in r:
            ranked.append((name, r["rmse"], r.get("r2", "-")))
        elif "titer_rmse" in r:
            ranked.append((name, r["titer_rmse"], "-"))
    ranked.sort(key=lambda x: x[1])

    for name, rmse, r2 in ranked:
        print(f"  {name:<15} {rmse:>10.4f}  {r2!s:>8}")
    for name, r in results.items():
        if "skipped" in r:
            print(f"  {name:<15}  (미학습)")
        elif "error" in r:
            print(f"  {name:<15}  오류")

    all_result_path = cfg.RESULTS_TT_DIR / "all_test_results.json"
    cfg.RESULTS_TT_DIR.mkdir(parents=True, exist_ok=True)
    with open(all_result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n전체 결과 저장: {all_result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모델 테스트")
    parser.add_argument("--model", type=str, required=True,
                        choices=ALL_MODELS + ["all"])
    parser.add_argument("--syn", action="store_true",
                        help="더미 데이터 사용")
    args = parser.parse_args()

    if args.model == "all":
        test_all(use_syn=args.syn)
    else:
        test_model(args.model, use_syn=args.syn)