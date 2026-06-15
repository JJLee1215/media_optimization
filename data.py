"""
data.py
실제 데이터 로딩 및 전처리

모델별 입력 형태에 맞게 데이터 반환:
  get_static_data()      → GP, XGBoost, RandomForest, MLP
  get_timeseries_data()  → RNN, LSTM, Transformer
  get_gnn_data()         → GNN (Model3)

실제 데이터 연결 시:
  DATA_PATH만 config.py에서 수정하면 됨
  각 함수 내부 컬럼명은 실제 데이터에 맞게 조정
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from config import Config

cfg = Config()


# ══════════════════════════════════════════════
# 1. 정적 모델용 데이터
#    GP, XGBoost, RandomForest, MLP
#    입력: (n_samples, n_features) 2D
# ══════════════════════════════════════════════

def get_static_data(use_syn: bool = False):
    """
    정적 모델용 데이터 로딩

    Args:
        use_syn: True → 더미 데이터 사용
                 False → 실제 데이터 사용 (config.DATA_STATIC)

    Returns:
        X_train, X_test  (n_samples, n_features)
        y_train, y_test  (n_samples,)
        x_cols           변수 이름 리스트
        scaler           StandardScaler (test 시 재사용)
    """
    # 경로 선택
    path = cfg.DATA_DIR / "batch_table_syn.csv" if use_syn else cfg.DATA_STATIC

    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} 없음. "
            f"더미 데이터는 python data_syn.py 실행, "
            f"실제 데이터는 {cfg.DATA_STATIC} 경로 확인"
        )

    df = pd.read_csv(path)

    # 컬럼명 처리 (원본: "Aeration rate(Fg:L/h)" → "Aeration rate")
    x_cols_full = cfg.STATIC_INPUT_COLS
    x_cols_short = [c.split("(")[0].strip() for c in x_cols_full]

    # 실제 존재하는 컬럼만 사용
    x_cols = [c for c in x_cols_short if c in df.columns]
    if not x_cols:
        # 원본 컬럼명으로 재시도
        x_cols = [c for c in x_cols_full if c in df.columns]

    if not x_cols:
        raise ValueError(f"입력 컬럼을 찾을 수 없음. 데이터 컬럼: {df.columns.tolist()}")

    X = df[x_cols].values.astype(np.float32)
    y = df[cfg.STATIC_TARGET_COL].values.astype(np.float32)

    print(f"[Static] 로드: {path}")
    print(f"  X: {X.shape}  Y: {y.shape}")
    print(f"  컬럼: {x_cols}")

    # train / test 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_SEED
    )

    # 스케일링
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"  train={len(X_train)}  test={len(X_test)}")
    return X_train, X_test, y_train, y_test, x_cols, scaler


# ══════════════════════════════════════════════
# 2. 시계열 모델용 데이터
#    RNN, LSTM, Transformer
#    입력: (n_samples, T, n_features) 3D
# ══════════════════════════════════════════════

def get_timeseries_data(use_syn: bool = False):
    """
    시계열 모델용 데이터 로딩

    Args:
        use_syn: True → 더미 데이터 사용

    Returns:
        loaders      {"train": DataLoader, "val": DataLoader}
        x_scaler     입력 스케일러
        y_scaler     출력 스케일러
        X_test       (n_test, T, n_features)
        y_test       (n_test,)
    """
    path = cfg.DATA_DIR / "timeseries_syn.csv" if use_syn else cfg.DATA_TIMESERIES

    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} 없음. "
            f"더미 데이터는 python data_syn.py 실행"
        )

    df = pd.read_csv(path)

    # Raman 컬럼 제거 (있으면)
    raman = [c for c in df.columns if c.startswith("R_Bin_")]
    df = df.drop(columns=raman)

    # Fault flag 0만 사용
    if cfg.TS_FAULT_COL in df.columns:
        df = df[df[cfg.TS_FAULT_COL] == 0]

    # 시퀀스 구성
    x_cols = [c.split("(")[0].strip() for c in cfg.TS_INPUT_COLS]
    x_cols = [c for c in x_cols if c in df.columns]
    if not x_cols:
        x_cols = [c for c in cfg.TS_INPUT_COLS if c in df.columns]

    X_list, y_list = [], []
    for _, grp in df.groupby(cfg.TS_BATCH_COL):
        grp = grp.sort_values(cfg.TS_TIME_COL)
        X_list.append(grp[x_cols].values)

        # 더미: 마지막 시점의 target, 실제: 각 배치의 최종 titer
        if cfg.TS_TARGET_COL in grp.columns:
            target_vals = grp[cfg.TS_TARGET_COL].values
            y_list.append(target_vals[target_vals > 0][-1] if any(target_vals > 0) else target_vals[-1])
        else:
            y_list.append(grp.iloc[-1][cfg.STATIC_TARGET_COL])

    seq_len = cfg.TS_SEQ_LEN
    X = np.array([x[:seq_len] for x in X_list], dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"[Timeseries] 로드: {path}")
    print(f"  X: {X.shape}  (배치수, 시점수, 변수수)")
    print(f"  Y: {y.shape}")

    # train / val / test 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=cfg.VAL_SIZE, random_state=cfg.RANDOM_SEED
    )
    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    # 스케일링
    n_feat = X.shape[2]
    x_scaler = StandardScaler()
    x_scaler.fit(X_train.reshape(-1, n_feat))
    X_train = x_scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_val   = x_scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
    X_test  = x_scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s   = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

    def to_loader(Xd, yd, shuffle=False, batch_size=16):
        from torch.utils.data import TensorDataset
        ds = TensorDataset(torch.tensor(Xd), torch.tensor(yd))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    loaders = {
        "train": to_loader(X_train, y_train_s, shuffle=True),
        "val":   to_loader(X_val, y_val_s),
    }

    return loaders, x_scaler, y_scaler, X_test, y_test


# ══════════════════════════════════════════════
# 3. GNN 모델용 데이터
#    Model3
#    입력: m_static (d_static,) + X_dynamic (T, d_dynamic)
# ══════════════════════════════════════════════

class GNNDataset(Dataset):
    """GNN 전용 Dataset"""
    def __init__(self, m_static, X_dynamic, y_titer, y_viab):
        self.m_static  = m_static    # (n, d_static)
        self.X_dynamic = X_dynamic   # (n, T, d_dynamic)
        self.y_titer   = y_titer     # (n,)
        self.y_viab    = y_viab      # (n,)

    def __len__(self):
        return len(self.m_static)

    def __getitem__(self, idx):
        return (
            self.m_static[idx],
            self.X_dynamic[idx],
            self.y_titer[idx],
            self.y_viab[idx],
        )


def get_gnn_data(use_syn: bool = False):
    """
    GNN 모델용 데이터 로딩

    Args:
        use_syn: True → 더미 데이터 사용

    Returns:
        train_loader, val_loader
    """
    if use_syn:
        static_path  = cfg.DATA_DIR / "gnn_m_static_syn.csv"
        dynamic_path = cfg.DATA_DIR / "gnn_X_dynamic_syn.csv"
    else:
        static_path  = cfg.DATA_GNN_STATIC
        dynamic_path = cfg.DATA_GNN_DYNAMIC

    for p in [static_path, dynamic_path]:
        if not Path(p).exists():
            raise FileNotFoundError(
                f"{p} 없음. "
                f"더미 데이터는 python data_syn.py 실행"
            )

    # 정적 데이터 로딩
    df_static = pd.read_csv(static_path)
    static_cols = ["glc0", "glut0", "Mn0", "Cu0"]
    m_static = torch.tensor(
        df_static[static_cols].values, dtype=torch.float32
    )
    y_titer = torch.tensor(df_static["y_titer"].values, dtype=torch.float32)
    y_viab  = torch.tensor(df_static["y_viab"].values,  dtype=torch.float32)

    # 동적 데이터 로딩
    df_dynamic = pd.read_csv(dynamic_path)
    dyn_cols = cfg.GNN_VARIABLE_NAMES
    n_batches = df_static.shape[0]
    T = cfg.GNN_T

    X_dynamic = torch.tensor(
        df_dynamic[dyn_cols].values.reshape(n_batches, T, cfg.GNN_D_DYNAMIC),
        dtype=torch.float32
    )

    print(f"[GNN] 로드: {static_path}, {dynamic_path}")
    print(f"  m_static  : {m_static.shape}")
    print(f"  X_dynamic : {X_dynamic.shape}")
    print(f"  y_titer   : {y_titer.shape}")

    # train / val 분리
    dataset = GNNDataset(m_static, X_dynamic, y_titer, y_viab)
    n_train = int(len(dataset) * cfg.TRAIN_RATIO)
    n_val   = len(dataset) - n_train

    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.RANDOM_SEED)
    )

    train_loader = DataLoader(
        train_set, batch_size=cfg.GNN_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.GNN_BATCH_SIZE, shuffle=False
    )

    print(f"  train={len(train_set)}  val={len(val_set)}")
    return train_loader, val_loader


# ══════════════════════════════════════════════
# 테스트
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("[1] 정적 데이터 로딩 테스트")
    print("=" * 50)
    X_train, X_test, y_train, y_test, x_cols, scaler = get_static_data(use_syn=True)
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

    print("\n" + "=" * 50)
    print("[2] 시계열 데이터 로딩 테스트")
    print("=" * 50)
    loaders, x_sc, y_sc, X_test_ts, y_test_ts = get_timeseries_data(use_syn=True)
    xb, yb = next(iter(loaders["train"]))
    print(f"  batch X: {xb.shape}  batch y: {yb.shape}")

    print("\n" + "=" * 50)
    print("[3] GNN 데이터 로딩 테스트")
    print("=" * 50)
    train_loader, val_loader = get_gnn_data(use_syn=True)
    m, X, yt, yv = next(iter(train_loader))
    print(f"  m_static : {m.shape}")
    print(f"  X_dynamic: {X.shape}")
    print(f"  y_titer  : {yt.shape}")
    print(f"  y_viab   : {yv.shape}")

    print("\n완료. 실제 데이터 연결 시 config.py 경로 수정")