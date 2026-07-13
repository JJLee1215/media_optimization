"""
training/result_schema.py
result.json의 selected_front_end / selected_back_end / matches 스키마 조립.

"무엇을 학습했나"(training/runners.py)와 별개로, "그 학습 조건이 프론트 요청과
실제로 일치했는가"를 검증하고 보고하는 역할만 담당.

스키마:
  selected_front_end : 프론트가 "요청"한 값 (selected_columns, pipeline, hyperparams=null)
  selected_back_end   : 실제로 "사용된" 값 (selected_columns, pipeline, hyperparams)
  matches              : front-end 요청과 back-end 실제값이 leaf 단위로 일치하는지
                          (true/false), 비교 대상이 없는 항목(hyperparams)은 null,
                          matches.overall은 전체 취합 결과
"""


def cols_match(requested, actual):
    """requested가 None이면(전체 사용 의도) 항상 True. 아니면 집합 비교."""
    if requested is None:
        return True
    return set(requested) == set(actual or [])


def build_result_schema(use_pipeline, selected_cols, selected_ts_cols,
                         embedding_model, other_blocks, notation,
                         actual_static_cols, actual_ts_cols,
                         actual_embedding_model, actual_pipeline_dim,
                         hyperparams):
    """
    selected_front_end / selected_back_end / matches 세 블록을 조립해서 반환.
    train.py의 train_model()이 basic_info와 합쳐 최종 result.json으로 저장함.
    """
    static_match = cols_match(selected_cols, actual_static_cols)
    ts_match     = cols_match(selected_ts_cols, actual_ts_cols)

    if use_pipeline:
        notation_match     = (notation or "smiles") == "smiles"   # 지금은 SMILES만 지원
        embedding_match     = embedding_model == actual_embedding_model
        other_blocks_match  = True   # other_blocks 필터링은 아직 미구현 — 항상 통과 (TODO)
        dim_match            = actual_pipeline_dim is not None
    else:
        notation_match = embedding_match = other_blocks_match = dim_match = True

    selected_front_end = {
        "input": {
            "selected_columns": {
                "static"     : selected_cols,
                "timeseries" : selected_ts_cols,
            }
        },
        "pipeline": {
            "enabled"      : use_pipeline,
            "notation"     : notation if use_pipeline else None,
            "embedding"    : embedding_model if use_pipeline else None,
            "other_blocks" : other_blocks if use_pipeline else None,
            "dim"          : None,
        },
        "hyperparams": None,
    }

    selected_back_end = {
        "input": {
            "selected_columns": {
                "static"     : actual_static_cols,
                "timeseries" : actual_ts_cols,
            }
        },
        "pipeline": {
            "enabled"      : use_pipeline,
            "notation"     : "smiles" if use_pipeline else None,
            "embedding"    : actual_embedding_model,
            "other_blocks" : other_blocks if use_pipeline else None,
            "dim"          : actual_pipeline_dim,
        },
        "hyperparams": hyperparams,
    }

    matches = {
        "input": {
            "selected_columns": {
                "static"     : static_match,
                "timeseries" : ts_match,
            }
        },
        "pipeline": {
            "enabled"      : True,
            "notation"     : notation_match,
            "embedding"    : embedding_match,
            "other_blocks" : other_blocks_match,
            "dim"          : dim_match,
        },
        "hyperparams": None,
    }

    leaf_values = [static_match, ts_match, notation_match, embedding_match, other_blocks_match, dim_match]
    matches["overall"] = all(leaf_values)

    return selected_front_end, selected_back_end, matches