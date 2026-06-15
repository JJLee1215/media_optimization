"""
1_data_check.py
IndPenSim 데이터 구조 탐색
- Raman(R_Bin_*) 제거
- 공정 변수 / Y 분리
- 배치별 시계열 구조 확인
- 결측치 / Fault 배치 확인
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless 환경
import matplotlib.pyplot as plt
from pathlib import Path

# ── 경로 ─────────────────────────────────────────────────────────────────────
DATA_PATH = "data_file/IndPenSim_Optimized_Final.csv"
OUT_DIR   = "outputs/eda"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ── 컬럼 그룹 정의 ───────────────────────────────────────────────────────────
TIME_COL  = "Time (h)"
BATCH_COL = "Batch ID"
FAULT_COL = "Fault flag"

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

STATE_COLS = [
    "Dissolved oxygen concentration(DO2:mg/L)",
    "pH(pH:pH)",
    "Temperature(T:K)",
    "Vessel Volume(V:L)",
    "Substrate concentration(S:g/L)",
    "carbon dioxide percent in off-gas(CO2outgas:%)",
    "Oxygen Uptake Rate(OUR:(g min^{-1}))",
    "Carbon evolution rate(CER:g/h)",
    "Generated heat(Q:kJ)",
    "Viscosity(Viscosity_offline:centPoise)",
]

OFFLINE_COLS = [
    "PAA concentration offline(PAA_offline:PAA (g L^{-1}))",
    "NH_3 concentration off-line(NH3_offline:NH3 (g L^{-1}))",
    "Offline Penicillin concentration(P_offline:P(g L^{-1}))",
    "Offline Biomass concentratio(X_offline:X(g L^{-1}))",
]

TARGET_COL = "Penicillin concentration(P:g/L)"


# ── 로드 ─────────────────────────────────────────────────────────────────────
def load(path: str) -> pd.DataFrame:
    print(f"[로드] {path}")
    df = pd.read_csv(path)

    raman_cols = [c for c in df.columns if c.startswith("R_Bin_")]
    df = df.drop(columns=raman_cols)
    print(f"  Raman {len(raman_cols)}개 제거  →  남은 컬럼: {len(df.columns)}개")
    return df


# ── 1. 기본 구조 ──────────────────────────────────────────────────────────────
def overview(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  1. 기본 구조")
    print("="*60)
    print(f"  전체 행: {len(df):,}")
    print(f"  전체 열: {len(df.columns)}")

    if BATCH_COL in df.columns:
        n = df[BATCH_COL].nunique()
        rows = df.groupby(BATCH_COL).size()
        print(f"  배치 수: {n}")
        print(f"  배치당 행: min={rows.min()}  max={rows.max()}  mean={rows.mean():.0f}")

    print(f"  시간 범위: {df[TIME_COL].min():.1f} ~ {df[TIME_COL].max():.1f} h")


# ── 2. 결측치 ─────────────────────────────────────────────────────────────────
def check_missing(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  2. 결측치")
    print("="*60)

    use_cols = [c for c in INPUT_COLS + STATE_COLS + OFFLINE_COLS if c in df.columns]
    miss = df[use_cols].isnull().sum()
    miss_pct = (miss / len(df) * 100).round(1)
    result = pd.DataFrame({"missing": miss, "pct(%)": miss_pct})
    result = result[result["missing"] > 0].sort_values("pct(%)", ascending=False)

    if result.empty:
        print("  결측치 없음")
    else:
        for col, row in result.iterrows():
            short = col.split("(")[0].strip()
            print(f"  {short:<35} {row['missing']:>6}행  ({row['pct(%)']}%)")


# ── 3. Fault 배치 ─────────────────────────────────────────────────────────────
def check_fault(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  3. Fault 배치")
    print("="*60)

    if FAULT_COL not in df.columns:
        print("  Fault flag 컬럼 없음")
        return

    fault_batches  = df[df[FAULT_COL] == 1][BATCH_COL].unique()
    normal_batches = df[df[FAULT_COL] == 0][BATCH_COL].unique()
    print(f"  정상 배치: {len(normal_batches)}개")
    print(f"  Fault 배치: {len(fault_batches)}개  → 학습 시 제외 권장")


# ── 4. Y 분포 ─────────────────────────────────────────────────────────────────
def check_target(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  4. Y — Penicillin 분포")
    print("="*60)

    if TARGET_COL not in df.columns:
        print("  컬럼 없음"); return

    # 배치별 최종값 (마지막 time point)
    normal_df = df[df[FAULT_COL] == 0] if FAULT_COL in df.columns else df
    final = normal_df.groupby(BATCH_COL)[TARGET_COL].last()

    print(f"  [배치별 최종 Penicillin (g/L)]")
    print(f"  min  : {final.min():.3f}")
    print(f"  mean : {final.mean():.3f}")
    print(f"  max  : {final.max():.3f}")
    print(f"  std  : {final.std():.3f}")
    return final


# ── 5. 컬럼 요약 ──────────────────────────────────────────────────────────────
def summarize_columns(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  5. 컬럼 그룹 요약")
    print("="*60)

    groups = {
        "입력(X) 조작 변수": INPUT_COLS,
        "상태 변수":          STATE_COLS,
        "오프라인 측정":      OFFLINE_COLS,
        "목표(Y)":            [TARGET_COL],
    }
    for name, cols in groups.items():
        exist = [c for c in cols if c in df.columns]
        print(f"\n  [{name}]  {len(exist)}개")
        for c in exist:
            print(f"    {c.split('(')[0].strip()}")


# ── 시각화 ───────────────────────────────────────────────────────────────────
def plot_trajectories(df: pd.DataFrame, n_sample: int = 15):
    normal_df = df[df[FAULT_COL] == 0] if FAULT_COL in df.columns else df
    batches   = normal_df[BATCH_COL].unique()[:n_sample]

    plot_cols = [TARGET_COL, "Sugar feed rate(Fs:L/h)",
                 "pH(pH:pH)", "Dissolved oxygen concentration(DO2:mg/L)"]
    plot_cols = [c for c in plot_cols if c in df.columns]

    fig, axes = plt.subplots(len(plot_cols), 1, figsize=(12, 3 * len(plot_cols)))
    if len(plot_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, plot_cols):
        for b in batches:
            sub = normal_df[normal_df[BATCH_COL] == b]
            ax.plot(sub[TIME_COL], sub[col], alpha=0.5, lw=0.8)
        ax.set_ylabel(col.split("(")[0].strip(), fontsize=8)
        ax.set_xlabel("Time (h)")

    fig.suptitle(f"배치 시계열 (정상 {len(batches)}개 샘플)", fontsize=11)
    plt.tight_layout()
    out = f"{OUT_DIR}/trajectories.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[그래프] {out}")


def plot_final_titer(df: pd.DataFrame):
    normal_df = df[df[FAULT_COL] == 0] if FAULT_COL in df.columns else df
    final = normal_df.groupby(BATCH_COL)[TARGET_COL].last()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(final, bins=20, edgecolor="white", color="#4B9FE0")
    ax.axvline(final.mean(), color="red", lw=1.5, ls="--",
               label=f"mean = {final.mean():.2f} g/L")
    ax.set_title("배치별 최종 Penicillin 분포 (정상 배치)")
    ax.set_xlabel("Final Penicillin (g/L)")
    ax.legend()
    plt.tight_layout()
    out = f"{OUT_DIR}/final_titer_dist.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[그래프] {out}")


def plot_correlation(df: pd.DataFrame):
    use_cols = [c for c in INPUT_COLS + STATE_COLS if c in df.columns]
    if TARGET_COL not in df.columns:
        return

    corr = df[use_cols + [TARGET_COL]].corr()[[TARGET_COL]].drop(TARGET_COL)
    corr = corr.sort_values(TARGET_COL)

    fig, ax = plt.subplots(figsize=(6, 7))
    colors = ["#E05C4B" if v < 0 else "#4B9FE0" for v in corr[TARGET_COL]]
    ax.barh(range(len(corr)), corr[TARGET_COL], color=colors)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels([c.split("(")[0].strip() for c in corr.index], fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Penicillin 상관계수")
    ax.set_xlabel("Pearson r")
    plt.tight_layout()
    out = f"{OUT_DIR}/correlation.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[그래프] {out}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load(DATA_PATH)

    overview(df)
    check_missing(df)
    check_fault(df)
    final = check_target(df)
    summarize_columns(df)

    print("\n[시각화 생성 중...]")
    plot_trajectories(df, n_sample=15)
    plot_final_titer(df)
    plot_correlation(df)

    # 정상 배치만 저장
    df_clean = df[df[FAULT_COL] == 0].copy() if FAULT_COL in df.columns else df.copy()
    out_csv = "data_file/pensim_clean.csv"
    df_clean.to_csv(out_csv, index=False)
    print(f"\n[저장] {out_csv}  ({len(df_clean):,}행 / {len(df_clean.columns)}열)")
    print("[완료] outputs/eda/ 폴더에서 그래프 확인하세요")