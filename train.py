"""
train.py
Model training entry point (CLI)

Usage:
  python train.py --model static
  python train.py --model gaussian_process --pipeline --embedding_model chemberta
  python train.py --model gaussian_process --pipeline --embedding_model rdkit --pooling_method multi_stat --use_pca --pca_dim 30
  python train.py --model xgboost --pipeline --embedding_model rdkit --other_blocks gem
  python train.py --model static --static_file my_batch.csv
  python train.py --model time --ts_file my_timeseries.csv
  python train.py --model all --static_file batch.csv --ts_file ts.csv

이 파일은 실행 진입점(CLI 파싱) + 오케스트레이션(train_model, train_group)만 담당.
실제 학습 로직은 training/runners.py, 결과 JSON 스키마 조립은
training/result_schema.py로 분리되어 있음.

※ result.json 최상위 구조:
  model, trained_at, data_file           ← basic_info에 해당하는 필드들
  selected_front_end / selected_back_end / matches
                                          ← training/result_schema.py 참고
  hyperparams                             ← 모델별 get_config() 결과
"""

import sys
import os
import argparse
import json
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from compare import collect_results, print_table, plot_comparison
from training.runners import (
    STATIC_MODELS, TIME_MODELS, STATIC_TIME_MODELS, ALL_MODELS, MODEL_GROUPS,
    train_static, train_time, train_static_time,
)
from training.result_schema import build_result_schema


def train_model(model_name, use_pipeline=False, static_file=None, ts_file=None,
                 selected_cols=None, selected_ts_cols=None,
                 embedding_model=None, other_blocks=None, notation=None,
                 pooling_method="mean", use_pca=False, pca_dim=30):
    config.make_dirs()
    print(f"\n{'='*55}")
    print(f"  Training    : {model_name.upper()}")
    print(f"  Pipeline    : {'ON' if use_pipeline else 'OFF'}"
          + (f"  (embedding_model={embedding_model}, pooling={pooling_method}, pca={use_pca}, other_blocks={other_blocks})" if use_pipeline else ""))
    print(f"  Static file : {static_file or config.DATA_STATIC}")
    print(f"  TS file     : {ts_file or config.DATA_TIMESERIES}")
    if selected_cols:
        print(f"  Selected static cols : {selected_cols}")
    if selected_ts_cols:
        print(f"  Selected ts cols     : {selected_ts_cols}")
    print(f"{'='*55}")

    if model_name in STATIC_MODELS:
        result = train_static(model_name, use_pipeline=use_pipeline, static_file=static_file,
                               selected_cols=selected_cols, embedding_model=embedding_model,
                               pooling_method=pooling_method, use_pca=use_pca, pca_dim=pca_dim,
                               other_blocks=other_blocks)
    elif model_name in TIME_MODELS:
        result = train_time(model_name, ts_file=ts_file, selected_cols=selected_ts_cols)
    elif model_name == "static_time_gnn":
        result = train_static_time(use_pipeline=use_pipeline, static_file=static_file, ts_file=ts_file,
                                    selected_cols=selected_cols, selected_ts_cols=selected_ts_cols,
                                    embedding_model=embedding_model,
                                    pooling_method=pooling_method, use_pca=use_pca, pca_dim=pca_dim,
                                    other_blocks=other_blocks)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    actual_static_cols     = result.pop("_actual_static_cols", None)
    actual_ts_cols          = result.pop("_actual_ts_cols", None)
    actual_embedding_model  = result.pop("_actual_embedding_model", None)
    actual_pipeline_dim     = result.pop("_actual_pipeline_dim", None)
    model_obj               = result.pop("_model_obj", None)

    hyperparams = model_obj.get_config() if (model_obj and hasattr(model_obj, "get_config")) else {}

    result["model"]      = model_name
    result["trained_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    result["data_file"]  = {
        "static"    : static_file or config.DATA_STATIC.name,
        "timeseries": ts_file or config.DATA_TIMESERIES.name,
    }

    selected_front_end, selected_back_end, matches = build_result_schema(
        use_pipeline            = use_pipeline,
        selected_cols           = selected_cols,
        selected_ts_cols        = selected_ts_cols,
        embedding_model         = embedding_model,
        other_blocks            = other_blocks,
        notation                = notation,
        actual_static_cols      = actual_static_cols,
        actual_ts_cols          = actual_ts_cols,
        actual_embedding_model  = actual_embedding_model,
        actual_pipeline_dim     = actual_pipeline_dim,
        hyperparams             = hyperparams,
        pooling_method           = pooling_method,
        use_pca                  = use_pca,
    )
    result["selected_front_end"] = selected_front_end
    result["selected_back_end"]  = selected_back_end
    result["matches"]            = matches

    result_path = config.result_dir(model_name) / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved: {result_path}  (matches.overall={matches['overall']})")
    return result


def train_group(group: str, use_pipeline=False, static_file=None, ts_file=None,
                 selected_cols=None, selected_ts_cols=None,
                 embedding_model=None, other_blocks=None, notation=None,
                 pooling_method="mean", use_pca=False, pca_dim=30):
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
                embedding_model=embedding_model,
                other_blocks=other_blocks,
                notation=notation,
                pooling_method=pooling_method,
                use_pca=use_pca,
                pca_dim=pca_dim,
            )
        except Exception as e:
            print(f"\n[{name}] Error: {e}")
            results[name] = {"error": str(e)}

    all_results = collect_results("train")
    print_table(all_results, "train")
    plot_comparison(all_results, "train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--model", type=str, required=True,
        choices=list(MODEL_GROUPS.keys()) + ALL_MODELS)
    parser.add_argument("--pipeline", action="store_true", default=False,
        help="Enable heterogeneity pipeline (static/static_time only)")
    parser.add_argument("--static_file", type=str, default=None)
    parser.add_argument("--ts_file", type=str, default=None)
    parser.add_argument("--selected_cols", type=str, default=None,
        help="쉼표로 구분된 static feature 컬럼명")
    parser.add_argument("--selected_ts_cols", type=str, default=None,
        help="쉼표로 구분된 timeseries feature 컬럼명")
    parser.add_argument("--embedding_model", type=str, default=None,
        choices=["rdkit", "chemberta", "unimol"],
        help="Heterogeneity pipeline 임베딩 종류 (--pipeline과 함께 사용)")
    parser.add_argument("--other_blocks", type=str, default=None,
        help="쉼표로 구분된 부가 블록 (log_conc,metal_physchem,gem). 없으면 전체 포함")
    parser.add_argument("--notation", type=str, default="smiles",
        help="분자 표기법 (현재는 smiles만 지원)")
    parser.add_argument("--pooling_method", type=str, default="mean",
        choices=["mean", "multi_stat"],
        help="Pooling 방식 (mean 또는 mean+weighted+max+count)")
    parser.add_argument("--use_pca", action="store_true", default=False,
        help="PCA로 최종 차원 압축 여부")
    parser.add_argument("--pca_dim", type=int, default=30,
        help="use_pca=True일 때 목표 차원")
    args = parser.parse_args()

    sel_cols    = args.selected_cols.split(",")    if args.selected_cols    else None
    sel_ts_cols = args.selected_ts_cols.split(",") if args.selected_ts_cols else None
    other_blks  = args.other_blocks.split(",")     if args.other_blocks     else None

    common_kwargs = dict(
        use_pipeline     = args.pipeline,
        static_file      = args.static_file,
        ts_file          = args.ts_file,
        selected_cols    = sel_cols,
        selected_ts_cols = sel_ts_cols,
        embedding_model  = args.embedding_model,
        other_blocks     = other_blks,
        notation         = args.notation,
        pooling_method    = args.pooling_method,
        use_pca           = args.use_pca,
        pca_dim           = args.pca_dim,
    )

    if args.model in MODEL_GROUPS:
        train_group(args.model, **common_kwargs)
    else:
        train_model(args.model, **common_kwargs)