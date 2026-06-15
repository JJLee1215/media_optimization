"""
data_syn.py
더미(합성) 데이터 생성

실제 데이터가 없을 때 구조 검증 및 코드 테스트용
실제 데이터 연결 시 data.py만 수정하면 됨

생성 데이터:
  1. 정적 모델용  (batch_table_syn.csv)
     - (n_samples, n_features) 2D
     - GP, XGBoost, RandomForest, MLP 용

  2. 시계열 모델용  (timeseries_syn.csv)
     - (배치수 × 시점수, 변수수) long format
     - RNN, LSTM, Transformer 용

  3. GNN 모델용
     - gnn_m_static_syn.csv    (n_batches, d_static)
     - gnn_X_dynamic_syn.csv   (n_batches × T, d_dynamic + 1)
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from config import Config

cfg = Config()

# ══════════════════════════════════════════════
# 1. 정적 모델용 더미 데이터
#    batch_table_syn.csv
# ══════════════════════════════════════════════

def make_static_data(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    GP, XGBoost, RandomForest, MLP 용
    입력: 공정 파라미터 평균값
    출력: titer_final
    """
    np.random.seed(seed)
    x_cols = [c.split("(")[0].strip() for c in cfg.STATIC_INPUT_COLS]

    data = {}
    ranges = {
        "Aeration rate"                       : (0.5,  2.0),
        "Agitator RPM"                        : (50,   200),
        "Sugar feed rate"                     : (0.01, 0.5),
        "Acid flow rate"                      : (0.0,  0.1),
        "Base flow rate"                      : (0.0,  0.1),
        "Heating/cooling water flow rate"     : (0.0,  5.0),
        "Heating water flow rate"             : (0.0,  2.0),
        "Water for injection/dilution"        : (0.0,  0.5),
        "PAA flow"                            : (0.0,  0.05),
        "Oil flow"                            : (0.0,  0.02),
    }

    for col, (lo, hi) in zip(x_cols, ranges.values()):
        data[col] = np.random.uniform(lo, hi, n_samples)

    df = pd.DataFrame(data)

    # titer_final: 공정 파라미터와 선형 관계 + 노이즈
    titer = (
        2.0 * df["Sugar feed rate"]
        + 0.5 * df["Aeration rate"]
        - 0.1 * df["Acid flow rate"]
        + np.random.normal(0, 0.3, n_samples)
    ).clip(0.1)

    df["titer_final"] = titer
    df.insert(0, "batch_id", range(1, n_samples + 1))

    save_path = cfg.DATA_DIR / "batch_table_syn.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[정적 데이터] 저장: {save_path}  shape: {df.shape}")
    return df


# ══════════════════════════════════════════════
# 2. 시계열 모델용 더미 데이터
#    timeseries_syn.csv (long format)
# ══════════════════════════════════════════════

def make_timeseries_data(
    n_batches: int = 100,
    seq_len: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """
    RNN, LSTM, Transformer 용
    long format: 각 행 = (배치, 시점)
    마지막 시점의 TARGET이 예측 대상
    """
    np.random.seed(seed)
    x_cols = [c.split("(")[0].strip() for c in cfg.TS_INPUT_COLS]
    rows = []

    for batch_id in range(1, n_batches + 1):
        # 배치별 기본값 샘플링
        base = np.random.uniform(0.5, 1.5, len(x_cols))
        titer_final = float(np.random.uniform(1.0, 5.0))

        for t in range(seq_len):
            row = {
                cfg.TS_BATCH_COL : batch_id,
                cfg.TS_TIME_COL  : t + 1,
                cfg.TS_FAULT_COL : 0,
            }
            for i, col in enumerate(x_cols):
                # 시간에 따라 천천히 변화 + 노이즈
                row[col] = max(0.0, base[i] + 0.005 * t + np.random.normal(0, 0.02))
            # 마지막 시점에만 titer 기록
            row[cfg.TS_TARGET_COL] = titer_final if t == seq_len - 1 else 0.0
            rows.append(row)

    df = pd.DataFrame(rows)

    save_path = cfg.DATA_DIR / "timeseries_syn.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[시계열 데이터] 저장: {save_path}  shape: {df.shape}")
    return df


# ══════════════════════════════════════════════
# 3. GNN 모델용 더미 데이터
#    gnn_m_static_syn.csv
#    gnn_X_dynamic_syn.csv
# ══════════════════════════════════════════════

def make_gnn_data(
    n_batches: int = None,
    T: int = None,
    seed: int = 42
) -> tuple:
    """
    GNN (Model3) 용
    m_static:  (n_batches, d_static)
    X_dynamic: (n_batches × T, d_dynamic + 2)  ← batch_id, time_step 포함
    """
    if n_batches is None:
        n_batches = cfg.GNN_N_BATCHES
    if T is None:
        T = cfg.GNN_T

    torch.manual_seed(seed)
    n = n_batches

    # ── m_static ─────────────────────────────
    m_static = torch.stack([
        torch.FloatTensor(n).uniform_(3.0, 6.0),    # glc0
        torch.FloatTensor(n).uniform_(1.0, 3.0),    # glut0
        torch.FloatTensor(n).uniform_(0.01, 0.1),   # Mn0
        torch.FloatTensor(n).uniform_(0.01, 0.05),  # Cu0
    ], dim=1)

    static_cols = ["glc0", "glut0", "Mn0", "Cu0"]
    df_static = pd.DataFrame(m_static.numpy(), columns=static_cols)
    df_static.insert(0, "batch_id", range(1, n + 1))

    # titer, viab 타깃 추가
    y_titer = (
        0.5 * m_static[:, 0]
        + 0.3 * m_static[:, 1]
        + torch.randn(n) * 0.2
    ).clamp(min=0.1)
    y_viab = (
        0.85 + 0.1 * m_static[:, 0] / 6.0
        + torch.randn(n) * 0.03
    ).clamp(0.5, 1.0)

    df_static["y_titer"] = y_titer.numpy()
    df_static["y_viab"]  = y_viab.numpy()

    # ── X_dynamic ────────────────────────────
    dyn_cols = cfg.GNN_VARIABLE_NAMES  # 9개
    rows = []

    for i in range(n):
        glc0 = m_static[i, 0].item()
        for t in range(T):
            feed_day = (t == 4) or (t == 8)
            row = {
                "batch_id"  : i + 1,
                "time_step" : t + 1,
                "glc"       : max(0.1, glc0 - 0.35*(t+1) + (1.5 if feed_day else 0) + np.random.normal(0, 0.1)),
                "glut"      : max(0.1, m_static[i,1].item() - 0.15*(t+1) + np.random.normal(0, 0.05)),
                "pH"        : 7.0 + np.random.normal(0, 0.1),
                "DO"        : max(10, 45 - 1.5*(t+1) + np.random.normal(0, 2)),
                "VCD"       : 0.5 + 0.5*(t+1) + np.random.normal(0, 0.1),
                "viab"      : max(0.5, 0.95 - 0.02*(t+1) + np.random.normal(0, 0.01)),
                "feed_glc"  : 2.0 if feed_day else 0.0,
                "feed_glut" : 0.5 if feed_day else 0.0,
                "feed_vol"  : 50.0 if feed_day else 0.0,
            }
            rows.append(row)

    df_dynamic = pd.DataFrame(rows)

    # 저장
    static_path  = cfg.DATA_DIR / "gnn_m_static_syn.csv"
    dynamic_path = cfg.DATA_DIR / "gnn_X_dynamic_syn.csv"
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_static.to_csv(static_path,  index=False)
    df_dynamic.to_csv(dynamic_path, index=False)

    print(f"[GNN 정적 데이터]   저장: {static_path}   shape: {df_static.shape}")
    print(f"[GNN 시계열 데이터] 저장: {dynamic_path}  shape: {df_dynamic.shape}")

    return df_static, df_dynamic


# ══════════════════════════════════════════════
# 메인: 전체 더미 데이터 한번에 생성
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("더미 데이터 생성")
    print("=" * 50)

    print("\n[1] 정적 모델용")
    df_static = make_static_data(n_samples=100)

    print("\n[2] 시계열 모델용")
    df_ts = make_timeseries_data(n_batches=100, seq_len=100)

    print("\n[3] GNN 모델용")
    df_gnn_s, df_gnn_d = make_gnn_data(n_batches=30, T=10)

    print("\n" + "=" * 50)
    print("완료. data_file/ 폴더 확인")
    print("실제 데이터 연결 시 data.py 수정")
    print("=" * 50)