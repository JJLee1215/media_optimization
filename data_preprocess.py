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

※ selected_cols 파라미터:
  UI Feature Selection에서 고른 컬럼명 리스트.
  None이면 (기존 동작대로) 전체 컬럼 사용.

※ embedding_model 파라미터 (신규):
  UI Heterogeneity 카드에서 고른 파이프라인 종류
  ("rdkit" | "chemberta" | "unimol" | None).
  get_static_data()가 이 값을 받아 get_pipeline()에 그대로 전달하고,
  실제로 어떤 파이프라인이 쓰였는지(actual_embedding_model)와
  그 차원 구성(pipeline_dim)을 함께 반환함.
  → train.py가 이 반환값을 이용해 "프론트에서 요청한 파이프라인"과
    "실제로 학습에 쓰인 파이프라인"이 일치하는지(matches) 검증하고
    result.json에 기록할 수 있게 됨.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

import config


# ══════════════════════════════════════════════
# 파이프라인 싱글톤 캐시
# ── RDKit/ChemBERTa/UniMol 셋 다 무거운 초기화(사전학습 모델 로딩 등)를
#    포함하므로, embedding_model 종류별로 한 번만 만들고 재사용.
#    (기존에는 RDKit 하나만 전역 싱글톤으로 캐싱했으나, 파이프라인이
#     3종류로 늘어나면서 dict 캐시로 확장함)
# ══════════════════════════════════════════════
_pipeline_cache = {}

# 각 파이프라인의 metal_physchem/gem 차원은 클래스 인스턴스가 아니라
# 각 heterogeneity/*.py 모듈의 최상단 상수로 정의되어 있음.
# get_pipeline_dim_info()에서 embedding_model에 맞는 모듈을 골라
# 이 상수들을 직접 가져와 dim 딕셔너리를 조립함.

def get_pipeline(embedding_model: str = "rdkit"):
    """
    embedding_model : "rdkit" | "chemberta" | "unimol"
                      None이면 "rdkit"으로 간주 (기존 동작과의 하위 호환)

    Returns: 해당 파이프라인 인스턴스
             (transform(X, feature_cols) 인터페이스는 세 클래스 모두 공통)
    """
    key = embedding_model or "rdkit"
    if key not in _pipeline_cache:
        if key == "chemberta":
            from heterogeneity.smile_BERTA_gem_pipe import ChemBERTaMediaPipeline
            _pipeline_cache[key] = ChemBERTaMediaPipeline()
        elif key == "unimol":
            from heterogeneity.smile_UniMol_gem_pipe import UniMolMediaPipeline
            _pipeline_cache[key] = UniMolMediaPipeline()
        else:
            from heterogeneity.smile_gem_pipe import MediaPipeline
            _pipeline_cache[key] = MediaPipeline()
    return _pipeline_cache[key]


def get_pipeline_dim_info(embedding_model: str, pipeline) -> dict:
    """
    실제로 초기화된 pipeline 인스턴스와 embedding_model 종류를 받아
    {embedding, metal_physchem, log_conc, gem, total} 딕셔너리를 조립.

    metal_physchem/gem 상수는 클래스 속성이 아니라 각 모듈의
    최상단 상수(METAL_PHYSCHEM_DIM, GEM_DIM)이므로, embedding_model에
    맞는 모듈을 직접 import해서 가져옴.
    """
    key = embedding_model or "rdkit"
    if key == "chemberta":
        from heterogeneity.smile_BERTA_gem_pipe import METAL_PHYSCHEM_DIM, GEM_DIM
    elif key == "unimol":
        from heterogeneity.smile_UniMol_gem_pipe import METAL_PHYSCHEM_DIM, GEM_DIM
    else:
        from heterogeneity.smile_gem_pipe import METAL_PHYSCHEM_DIM, GEM_DIM

    return {
        "embedding"      : pipeline._emb_dim,
        "metal_physchem" : METAL_PHYSCHEM_DIM,
        "log_conc"       : 1,
        "gem"            : GEM_DIM,
        "total"          : pipeline.vector_dim,
    }


def get_static_data(use_pipeline: bool = False, static_file: str = None,
                     selected_cols: list = None, embedding_model: str = None):
    """
    Load and preprocess static data for GP, XGBoost, RandomForest, MLP.

    static_file     : 파일명 (예: "batch_table_syn.csv")
                       None이면 config.DATA_STATIC 기본값 사용
                       UI에서 업로드한 파일명을 전달하면 해당 파일로 학습

    use_pipeline    : True → heterogeneity pipeline 적용 (차원은 embedding_model에 따라 다름)
                       False → raw features 그대로 (기본값)

    selected_cols   : 사용할 feature 컬럼명 리스트 (예: ["Glucose_0", "Glutamine_0"])
                       None이면 전체 컬럼 사용 (기존 동작)

    embedding_model : "rdkit" | "chemberta" | "unimol" | None
                       use_pipeline=True일 때만 의미 있음.
                       None이면 "rdkit"으로 처리됨 (하위 호환).

    Returns:
        X_train                 (n_train, VECTOR_DIM) or (n_train, len(x_cols))
        X_test                  (n_test,  VECTOR_DIM) or (n_test,  len(x_cols))
        y_train                 (n_train,)
        y_test                  (n_test,)
        x_cols                  list of feature names
        scaler                  fitted StandardScaler
        actual_embedding_model  실제로 사용된 파이프라인 종류 (use_pipeline=False면 None)
        pipeline_dim            {"embedding":.., "metal_physchem":.., "log_conc":1,
                                  "gem":.., "total":..} (use_pipeline=False면 None)
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

    # ── 선택된 컬럼만 사용 ──
    if selected_cols:
        x_cols = [c for c in x_cols if c in selected_cols]
        if not x_cols:
            raise ValueError("selected_cols와 일치하는 static feature가 없습니다.")

    X = df[x_cols].values.astype(np.float32)
    y = df["titer_final"].values.astype(np.float32)

    print(f"[preprocess] Loaded: {file_path}")
    print(f"  Raw X: {X.shape}  y: {y.shape}")
    print(f"  Features: {x_cols}")

    # ── Step 1~4: media pipeline ──────────────
    # 실제로 어떤 파이프라인이 쓰였는지(actual_embedding_model)와
    # 그 차원 구성(pipeline_dim)을 여기서 확정해 반환값에 실어 보냄.
    # → train.py가 프론트 요청값(requested)과 이 실측값(actual)을
    #   비교해서 result.json의 matches를 계산할 수 있게 됨.
    actual_embedding_model = None
    pipeline_dim            = None
    if use_pipeline:
        pipeline = get_pipeline(embedding_model)
        actual_embedding_model = embedding_model or "rdkit"
        X = pipeline.transform(X, x_cols)
        pipeline_dim = get_pipeline_dim_info(actual_embedding_model, pipeline)
        print(f"  Pipeline X: {X.shape}  (embedding_model={actual_embedding_model})")

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
    return X_train, X_test, y_train, y_test, x_cols, scaler, actual_embedding_model, pipeline_dim


def get_timeseries_data(batch_size=None, ts_file: str = None, selected_cols: list = None):
    """
    Load and preprocess time series data for RNN, LSTM, Transformer.

    ts_file       : 파일명 (예: "timeseries_syn.csv")
                    None이면 config.DATA_TIMESERIES 기본값 사용
                    UI에서 업로드한 파일명을 전달하면 해당 파일로 학습

    selected_cols : 사용할 timeseries feature 컬럼명 리스트
                    None이면 전체 컬럼 사용 (기존 동작)

    ※ 시계열 pipeline 적용은 추후 고려, 현재는 기존 동작 유지
      (static과 달리 embedding_model 파라미터가 없음)

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

    # ── 선택된 컬럼만 사용 ──
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
            titer_vals[titer_vals > 0][-1]
            if any(titer_vals > 0) else titer_vals[-1]
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
    print("[1] Static data (pipeline=False, 기본값)")
    X_train, X_test, y_train, y_test, x_cols, scaler, emb_model, dim_info = get_static_data()
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    print(f"  embedding_model: {emb_model}  dim: {dim_info}")   # 둘 다 None이어야 정상

    print("\n[2] Static data (pipeline=True, embedding_model=rdkit)")
    X_train2, X_test2, _, _, _, _, emb_model2, dim_info2 = get_static_data(
        use_pipeline=True, embedding_model="rdkit"
    )
    print(f"  X_train: {X_train2.shape}  X_test: {X_test2.shape}")
    print(f"  embedding_model: {emb_model2}  dim: {dim_info2}")

    print("\n[3] Timeseries data")
    loaders, x_sc, y_sc, X_test_ts, y_test_ts, feat_cols = get_timeseries_data()
    xb, yb = next(iter(loaders["train"]))
    print(f"  batch X: {xb.shape}  batch y: {yb.shape}")

    print("\n[4] Static data (selected_cols 테스트)")
    X_train3, X_test3, _, _, x_cols3, _, _, _ = get_static_data(
        selected_cols=["Glucose_0", "Glutamine_0"]
    )
    print(f"  X_train: {X_train3.shape}  x_cols: {x_cols3}")