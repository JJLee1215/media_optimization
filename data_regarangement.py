"""
2_make_batch_table.py
시계열 데이터 → 배치 단위 테이블 변환

각 배치의 첫 행  → X (초기 공정 조건)
각 배치의 마지막 행 → Y (최종 Penicillin titer)
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = "data_file/IndPenSim_Optimized_Final.csv"
OUT_PATH  = "data_file/batch_table.csv"
OUT_DIR   = "outputs/eda"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

BATCH_COL = "Batch ID"
FAULT_COL = "Fault flag"
TIME_COL  = "Time (h)"
TARGET    = "Penicillin concentration(P:g/L)"

INPUT_COLS = [
    "Aeration rate(Fg:L/h)",
    "Agitator RPM(RPM:RPM)",
    "Sugar feed rate(Fs:L/h)",
    "Acid flow rate(Fa:L/h)",
    "Base flow rate(Fb:L/h)",
    "Heating/cooling water flow rate(Fc:L/h)",
    "Heating water flow rate(Fh:L/h)",
    "Water for injection/dilution(Fw:L/h)",
    "PAA flow(Fpaa:PAA flow (L/h))",
    "Oil flow(Foil:L/hr)",
]

SHORT = {c: c.split("(")[0].strip() for c in INPUT_COLS}


def load(path):
    df = pd.read_csv(path)
    raman = [c for c in df.columns if c.startswith("R_Bin_")]
    df = df.drop(columns=raman)
    # 정상 배치만
    if FAULT_COL in df.columns:
        before = df[BATCH_COL].nunique()
        df = df[df[FAULT_COL] == 0]
        after  = df[BATCH_COL].nunique()
        print(f"  Fault 배치 제거: {before} → {after}개")
    return df


def make_batch_table(df):
    """
    배치별로
      첫 행  → X (초기 조건)
      마지막 행 → Y (최종 titer)
    """
    records = []
    exist_inputs = [c for c in INPUT_COLS if c in df.columns]

    for batch_id, grp in df.groupby(BATCH_COL):
        grp = grp.sort_values(TIME_COL)

        first = grp.iloc[0]   # 첫 시점
        last  = grp.iloc[-1]  # 마지막 시점

        row = {"batch_id": batch_id}

        # X: 초기값
        for col in exist_inputs:
            row[SHORT[col]] = first[col]

        # Y: 최종 titer
        row["titer_final"] = last[TARGET]

        records.append(row)

    return pd.DataFrame(records)


def check_x_variation(batch_df):
    """
    배치마다 초기 X값이 얼마나 다른지 확인
    변동계수(CV = std/mean)가 크면 → 배치마다 다름 → X로 쓸 가치 있음
    """
    print("\n" + "="*60)
    print("  배치 간 초기값 변동 확인  (CV = std/mean)")
    print("  CV > 0.1 이면 배치마다 의미있게 다름")
    print("="*60)

    x_cols = [c for c in batch_df.columns if c not in ["batch_id", "titer_final"]]

    print(f"\n{'변수':<30} {'mean':>10} {'std':>10} {'CV':>8}  판정")
    print("-"*65)
    for col in x_cols:
        s    = batch_df[col]
        mean = s.mean()
        std  = s.std()
        cv   = std / (abs(mean) + 1e-9)
        flag = "✓ 변동 있음" if cv > 0.05 else "  거의 고정"
        print(f"{col:<30} {mean:>10.4f} {std:>10.4f} {cv:>8.3f}  {flag}")


def check_y_variation(batch_df):
    print("\n" + "="*60)
    print("  Y (최종 titer) 분포")
    print("="*60)
    y = batch_df["titer_final"]
    print(f"  배치 수 : {len(y)}")
    print(f"  min     : {y.min():.3f}")
    print(f"  mean    : {y.mean():.3f}")
    print(f"  max     : {y.max():.3f}")
    print(f"  std     : {y.std():.3f}")
    print(f"  CV      : {y.std()/y.mean():.3f}")

    if y.std() / y.mean() < 0.05:
        print("\n  ⚠ Y 변동이 매우 작음 → 배치 간 차이가 없을 수 있음")
    else:
        print("\n  ✓ Y 변동 충분 → 예측 모델 학습 가능")


if __name__ == "__main__":
    print("[로드]", DATA_PATH)
    df = load(DATA_PATH)

    print("\n[변환] 시계열 → 배치 테이블")
    batch_df = make_batch_table(df)

    print(f"\n[결과]")
    print(f"  행(배치 수): {len(batch_df)}")
    print(f"  열(변수 수): {len(batch_df.columns)}")
    print(f"\n[미리보기 (5행)]")
    print(batch_df.head().to_string(index=False))

    check_x_variation(batch_df)
    check_y_variation(batch_df)

    batch_df.to_csv(OUT_PATH, index=False)
    print(f"\n[저장] {OUT_PATH}")
    print("[완료] 다음: python scripts/3_model.py")