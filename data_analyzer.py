"""
data_analyzer.py
Static data analysis — all outputs saved as PNG

Functions:
  basic_stats()           → JSON  기초 통계
  missing_analysis()      → JSON  결측치 분석
  correlation_heatmap()   → PNG   컴포넌트 간 상관관계 히트맵
  correlation_stats()     → PNG   Pearson/Spearman/p-value vs Titer  ★ 신규
  distribution_plots()    → PNG   컴포넌트별 히스토그램
  titer_correlation_plots() → PNG 컴포넌트 vs Titer 산점도
  outlier_plots()         → PNG   아웃라이어 탐지 (IQR)              ★ 신규
  pca_plots()             → PNG   PCA biplot + 설명분산               ★ 신규
  timeseries_profile()    → PNG   시계열 평균 프로파일
  run_all()               전부 실행

탭별 사용:
  Overview     : basic_stats, missing_analysis
  Correlation  : correlation_heatmap, correlation_stats
  Distribution : distribution_plots, outlier_plots
  Titer        : titer_correlation_plots
  PCA          : pca_plots
  Timeseries   : timeseries_profile
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, str(Path(__file__).parent))
import config

RESULT_DIR = config.RESULTS_TT_DIR / "data_analysis"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MEDIA_COLS  = ["Glucose_0", "Glutamine_0", "Asparagine_0",
               "Lactate_0", "Ammonia_0", "Cu_0", "Zn_0", "Mn_0", "Fe_0"]
TARGET_COL  = "titer_final"

COLORS = ["#1D9E75","#534AB7","#E24B4A","#EF9F27","#185FA5",
          "#9FE1CB","#AFA9EC","#F0997B","#888780"]


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════

def _load(filepath: str = None) -> pd.DataFrame:
    path = Path(filepath) if filepath else config.DATA_STATIC
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def _media_cols(df: pd.DataFrame) -> list:
    return [c for c in MEDIA_COLS if c in df.columns]


# ══════════════════════════════════════════════
# Overview
# ══════════════════════════════════════════════

def basic_stats(filepath: str = None, batch_id=None) -> dict:
    """
    기초 통계 — mean, std, min, max, 25/50/75 percentile.

    batch_id = None or "all" : 전체 배치 평균 (describe)
    batch_id = 특정 번호     : 해당 배치의 실제 조성값 반환
                               (static 데이터는 배치당 행이 1개이므로
                                describe 대신 실제값을 mean 키로 반환)
    """
    df   = _load(filepath)
    cols = _media_cols(df)

    # ── 특정 배치 선택 ──
    if batch_id and str(batch_id) != "all" and "Batch_ID" in df.columns:
        row = df[df["Batch_ID"] == int(batch_id)]
        if len(row) == 0:
            raise ValueError(f"Batch {batch_id} not found.")

        result = {}
        all_cols = cols + ([TARGET_COL] if TARGET_COL in df.columns else [])
        for col in all_cols:
            if col not in row.columns:
                continue
            val = float(row[col].values[0])
            result[col] = {
                "count": 1, "mean": val, "std": 0.0,
                "min": val, "25%": val, "50%": val,
                "75%": val, "max": val,
            }
        return result

    # ── 전체 평균 (All) ──
    desc = df[cols + ([TARGET_COL] if TARGET_COL in df.columns else [])].describe()
    return desc.round(4).to_dict()


def missing_analysis(filepath: str = None) -> dict:
    """결측치 개수 및 비율."""
    df      = _load(filepath)
    cols    = _media_cols(df)
    missing = df[cols].isnull().sum()
    ratio   = (missing / len(df) * 100).round(2)
    return {
        "missing_count": missing.to_dict(),
        "missing_ratio": ratio.to_dict(),
        "total_rows"   : len(df),
    }


# ══════════════════════════════════════════════
# Correlation
# ══════════════════════════════════════════════

def correlation_heatmap(filepath: str = None,
                        selected_cols: list = None,
                        batch_id=None) -> str:
    """컴포넌트 간 Pearson 상관관계 히트맵 → PNG."""
    df   = _load(filepath)
    cols = selected_cols if selected_cols else _media_cols(df)
    if batch_id and batch_id != "all" and "Batch_ID" in df.columns:
        df = df[df["Batch_ID"] == int(batch_id)]

    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5,
                annot_kws={"size": 9})
    ax.set_title("Component correlation heatmap", fontsize=13, pad=12)
    plt.tight_layout()
    out = RESULT_DIR / "correlation_heatmap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return str(out)


def correlation_stats(filepath: str = None,
                      selected_cols: list = None) -> str:
    """
    Pearson / Spearman / p-value vs Titer 테이블 → PNG.
    ★ 신규 — XAI PCC/Spearman 사전 검증용
    유의수준: *** p<0.001  ** p<0.01  * p<0.05  ns p≥0.05
    """
    df   = _load(filepath)
    cols = selected_cols if selected_cols else _media_cols(df)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    y = df[TARGET_COL].values
    rows = []
    for col in cols:
        x = df[col].values
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        sig = ("***" if pp < 0.001 else
               "**"  if pp < 0.01  else
               "*"   if pp < 0.05  else "ns")
        rows.append({
            "Component": col,
            "Pearson r" : round(pr, 3),
            "Pearson p" : round(pp, 4),
            "Spearman r": round(sr, 3),
            "Spearman p": round(sp, 4),
            "Sig."      : sig,
        })

    result_df = pd.DataFrame(rows).sort_values("Pearson r",
                                                ascending=False,
                                                key=abs)

    # 테이블을 PNG로 렌더링
    fig, ax = plt.subplots(figsize=(10, len(rows) * 0.55 + 1.2))
    ax.axis("off")
    tbl = ax.table(
        cellText  = result_df.values,
        colLabels = result_df.columns,
        cellLoc   = "center",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)

    # 헤더 색상
    for j in range(len(result_df.columns)):
        tbl[0, j].set_facecolor("#1a1f2e")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # 유의미한 행 강조 (p < 0.05)
    for i, row in enumerate(rows):
        if row["Sig."] in ("*", "**", "***"):
            for j in range(len(result_df.columns)):
                tbl[i + 1, j].set_facecolor("#E1F5EE")

    ax.set_title("Correlation with Titer (Pearson / Spearman / p-value)",
                 fontsize=12, pad=12, fontweight="bold")
    plt.tight_layout()
    out = RESULT_DIR / "correlation_stats.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return str(out)


# ══════════════════════════════════════════════
# Distribution
# ══════════════════════════════════════════════

def distribution_plots(filepath: str = None,
                       selected_cols: list = None,
                       batch_id=None) -> str:
    """컴포넌트별 히스토그램 → PNG."""
    df   = _load(filepath)
    cols = selected_cols if selected_cols else _media_cols(df)
    if batch_id and batch_id != "all" and "Batch_ID" in df.columns:
        df = df[df["Batch_ID"] == int(batch_id)]

    n    = len(cols)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol,
                              figsize=(ncol * 4, nrow * 3))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=15,
                     color=COLORS[i % len(COLORS)],
                     edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=13, y=1.01)
    plt.tight_layout()
    out = RESULT_DIR / "distribution.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return str(out)


def outlier_plots(filepath: str = None,
                  selected_cols: list = None) -> str:
    """
    IQR 기반 아웃라이어 탐지 → boxplot PNG.
    ★ 신규 — bee swarm 사전 버전
    """
    df   = _load(filepath)
    cols = selected_cols if selected_cols else _media_cols(df)

    fig, ax = plt.subplots(figsize=(len(cols) * 1.2 + 2, 5))
    data = [df[c].dropna().values for c in cols]
    bp   = ax.boxplot(data, patch_artist=True, notch=False,
                      medianprops={"color": "white", "linewidth": 2})

    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, len(cols) + 1))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=9)
    ax.set_title("Outlier detection (IQR boxplot)", fontsize=13)
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = RESULT_DIR / "outlier.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return str(out)


# ══════════════════════════════════════════════
# Titer analysis
# ══════════════════════════════════════════════

def titer_correlation_plots(filepath: str = None,
                             selected_cols: list = None) -> str:
    """컴포넌트 vs Titer 산점도 (회귀선 포함) → PNG."""
    df   = _load(filepath)
    cols = selected_cols if selected_cols else _media_cols(df)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    n    = len(cols)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol,
                              figsize=(ncol * 4, nrow * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        x = df[col].values
        y = df[TARGET_COL].values
        pr, pp = stats.pearsonr(x, y)
        axes[i].scatter(x, y, alpha=0.5, s=20,
                        color=COLORS[i % len(COLORS)])
        # 회귀선
        m, b = np.polyfit(x, y, 1)
        xs   = np.linspace(x.min(), x.max(), 100)
        axes[i].plot(xs, m * xs + b, "r--", lw=1.2)
        axes[i].set_xlabel(col, fontsize=9)
        axes[i].set_ylabel("Titer (g/L)", fontsize=9)
        sig  = ("***" if pp < 0.001 else
                "**"  if pp < 0.01  else
                "*"   if pp < 0.05  else "ns")
        axes[i].set_title(f"r={pr:.2f} {sig}", fontsize=10)
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Component vs Titer", fontsize=13, y=1.01)
    plt.tight_layout()
    out = RESULT_DIR / "titer_correlation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return str(out)


# ══════════════════════════════════════════════
# PCA
# ══════════════════════════════════════════════

def pca_plots(filepath: str = None,
              selected_cols: list = None) -> str:
    """
    PCA biplot + 설명분산 bar chart → PNG.
    ★ 신규 — feature 중요도 힌트, XAI 사전 검증용
    """
    df   = _load(filepath)
    cols = selected_cols if selected_cols else _media_cols(df)

    X      = df[cols].dropna().values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    pca    = PCA(n_components=min(len(cols), X_sc.shape[0]))
    scores = pca.fit_transform(X_sc)
    loads  = pca.components_

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Biplot (PC1 vs PC2) ──
    ax = axes[0]
    # 색상: titer가 있으면 titer로 컬러링
    if TARGET_COL in df.columns:
        titer = df[cols + [TARGET_COL]].dropna()[TARGET_COL].values
        sc = ax.scatter(scores[:, 0], scores[:, 1],
                        c=titer, cmap="RdYlGn", alpha=0.7, s=30)
        fig.colorbar(sc, ax=ax, label="Titer (g/L)")
    else:
        ax.scatter(scores[:, 0], scores[:, 1],
                   color="#1D9E75", alpha=0.7, s=30)

    # Loading vectors
    scale = 3.0
    for j, col in enumerate(cols):
        ax.arrow(0, 0, loads[0, j] * scale, loads[1, j] * scale,
                 head_width=0.08, head_length=0.05,
                 fc="#534AB7", ec="#534AB7", alpha=0.8)
        ax.text(loads[0, j] * scale * 1.15,
                loads[1, j] * scale * 1.15,
                col.replace("_0", ""), fontsize=8, color="#534AB7",
                ha="center", va="center")

    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=10)
    ax.set_title("PCA biplot", fontsize=12)
    ax.grid(True, alpha=0.3)

    # ── 설명분산 bar chart ──
    ax2    = axes[1]
    n_comp = min(len(cols), 9)
    evr    = pca.explained_variance_ratio_[:n_comp] * 100
    cumsum = np.cumsum(evr)
    x_pos  = np.arange(1, n_comp + 1)

    bars = ax2.bar(x_pos, evr, color="#1D9E75", alpha=0.8, edgecolor="white")
    ax2.plot(x_pos, cumsum, "o-", color="#E24B4A", lw=1.5,
             ms=5, label="Cumulative")
    ax2.axhline(80, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax2.text(n_comp - 0.3, 81, "80%", fontsize=8, color="gray")
    ax2.set_xlabel("Principal component", fontsize=10)
    ax2.set_ylabel("Explained variance (%)", fontsize=10)
    ax2.set_title("Explained variance", fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = RESULT_DIR / "pca.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return str(out)


# ══════════════════════════════════════════════
# Timeseries
# ══════════════════════════════════════════════

def timeseries_profile(filepath: str = None) -> str:
    """14일 평균 농도 프로파일 → PNG."""
    path = Path(filepath) if filepath else config.DATA_TIMESERIES
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df        = pd.read_csv(path)
    time_col  = "Time (day)"
    skip_cols = ["Batch_ID", time_col, "Fault flag", "Titer (g/L)",
                 "Viability", "VCD"]
    feat_cols = [c for c in df.columns if c not in skip_cols
                 and df[c].dtype in [float, int]][:9]

    mean_df = df.groupby(time_col)[feat_cols].mean()

    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    axes = axes.flatten()

    for i, col in enumerate(feat_cols):
        axes[i].plot(mean_df.index, mean_df[col],
                     color=COLORS[i % len(COLORS)], lw=2)
        axes[i].fill_between(mean_df.index, mean_df[col],
                              alpha=0.15, color=COLORS[i % len(COLORS)])
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("Day")
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("14-day mean profile (timeseries)", fontsize=13)
    plt.tight_layout()
    out = RESULT_DIR / "timeseries_profile.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return str(out)


# ══════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════

def run_all(filepath: str = None) -> dict:
    results = {}
    fns = [
        ("basic_stats",              lambda: basic_stats(filepath)),
        ("missing_analysis",         lambda: missing_analysis(filepath)),
        ("correlation_heatmap",      lambda: correlation_heatmap(filepath)),
        ("correlation_stats",        lambda: correlation_stats(filepath)),
        ("distribution_plots",       lambda: distribution_plots(filepath)),
        ("outlier_plots",            lambda: outlier_plots(filepath)),
        ("titer_correlation_plots",  lambda: titer_correlation_plots(filepath)),
        ("pca_plots",                lambda: pca_plots(filepath)),
        ("timeseries_profile",       lambda: timeseries_profile()),
    ]
    for name, fn in fns:
        try:
            results[name] = fn()
            print(f"  [{name}] done")
        except Exception as e:
            results[name] = f"error: {e}"
            print(f"  [{name}] error: {e}")
    return results


if __name__ == "__main__":
    print("Running all analyses...")
    run_all()
    print("Done →", RESULT_DIR)


# ══════════════════════════════════════════════
# Timeseries 전용 분석 (신규)
# ══════════════════════════════════════════════

def batch_overlay(filepath: str = None) -> str:
    """
    배치별 시계열 오버레이 — 각 배치의 주요 성분 농도 변화를 겹쳐서 시각화.
    배치 간 변동성 파악용.
    ★ 신규
    """
    path = Path(filepath) if filepath else config.DATA_TIMESERIES
    df   = pd.read_csv(path)

    if "Fault flag" in df.columns:
        df = df[df["Fault flag"] == 0]

    batch_col = "Batch_ID"
    time_col  = "Time (day)"
    plot_cols = ["Glucose_conc", "Glutamine_conc", "Asparagine_conc",
                 "Lactate_conc", "Ammonia_conc"]
    plot_cols = [c for c in plot_cols if c in df.columns][:4]

    n    = len(plot_cols)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))
    if n == 1:
        axes = [axes]

    batches = sorted(df[batch_col].unique())

    for i, col in enumerate(plot_cols):
        for bid in batches:
            grp = df[df[batch_col] == bid].sort_values(time_col)
            axes[i].plot(grp[time_col], grp[col],
                         alpha=0.2, lw=0.8, color=COLORS[i % len(COLORS)])
        # 평균선
        mean_df = df.groupby(time_col)[col].mean()
        axes[i].plot(mean_df.index, mean_df.values,
                     color=COLORS[i % len(COLORS)], lw=2.5, label="Mean")
        axes[i].set_title(col.replace("_conc", ""), fontsize=11)
        axes[i].set_xlabel("Day")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)

    fig.suptitle("Batch overlay (all batches + mean)", fontsize=13)
    plt.tight_layout()
    out = RESULT_DIR / "batch_overlay.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return str(out)


def titer_trajectory(filepath: str = None) -> str:
    """
    Titer trajectory — 배치별 titer 시간 변화.
    최종 titer 기준으로 상위/하위 배치 구분해서 시각화.
    ★ 신규
    """
    path = Path(filepath) if filepath else config.DATA_TIMESERIES
    df   = pd.read_csv(path)

    batch_col  = "Batch_ID"
    time_col   = "Time (day)"
    titer_col  = "Titer (g/L)"

    if titer_col not in df.columns:
        raise ValueError(f"'{titer_col}' column not found.")

    # 최종 titer 계산
    final_titer = (
        df[df[titer_col] > 0]
        .groupby(batch_col)[titer_col]
        .last()
        .sort_values(ascending=False)
    )
    top_n    = 10
    top_ids  = final_titer.index[:top_n].tolist()
    bot_ids  = final_titer.index[-top_n:].tolist()

    fig, ax = plt.subplots(figsize=(11, 5))

    for bid in df[batch_col].unique():
        grp = df[df[batch_col] == bid].sort_values(time_col)
        if bid in top_ids:
            ax.plot(grp[time_col], grp[titer_col],
                    color="#1D9E75", alpha=0.6, lw=1.2)
        elif bid in bot_ids:
            ax.plot(grp[time_col], grp[titer_col],
                    color="#E24B4A", alpha=0.6, lw=1.2)
        else:
            ax.plot(grp[time_col], grp[titer_col],
                    color="#888780", alpha=0.15, lw=0.8)

    # 범례용 더미
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#1D9E75", lw=2, label=f"Top {top_n} titer"),
        Line2D([0], [0], color="#E24B4A", lw=2, label=f"Bottom {top_n} titer"),
        Line2D([0], [0], color="#888780", lw=1, alpha=0.5, label="Others"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_xlabel("Day")
    ax.set_ylabel("Titer (g/L)")
    ax.set_title("Titer trajectory — Top vs Bottom batches", fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULT_DIR / "titer_trajectory.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return str(out)


def ts_correlation(filepath: str = None) -> str:
    """
    Day별 컴포넌트 vs 최종 Titer 상관관계 히트맵.
    "어느 날의 어떤 성분이 최종 titer와 가장 관련있는가"
    XAI 결과 사전 검증용.
    ★ 신규
    """
    path = Path(filepath) if filepath else config.DATA_TIMESERIES
    df   = pd.read_csv(path)

    if "Fault flag" in df.columns:
        df = df[df["Fault flag"] == 0]

    batch_col  = "Batch_ID"
    time_col   = "Time (day)"
    titer_col  = "Titer (g/L)"

    conc_cols = ["Glucose_conc", "Glutamine_conc", "Asparagine_conc",
                 "Lactate_conc", "Ammonia_conc", "Cu_conc", "Zn_conc",
                 "Mn_conc", "Fe_conc"]
    conc_cols = [c for c in conc_cols if c in df.columns]

    # 최종 titer
    final_titer = (
        df[df[titer_col] > 0]
        .groupby(batch_col)[titer_col]
        .last()
    )

    days   = sorted(df[time_col].unique())
    corr_matrix = pd.DataFrame(index=conc_cols, columns=days, dtype=float)

    for day in days:
        day_df = df[df[time_col] == day].set_index(batch_col)
        for col in conc_cols:
            if col not in day_df.columns:
                continue
            merged = pd.concat([day_df[col], final_titer], axis=1).dropna()
            if len(merged) < 5:
                continue
            r, _ = stats.pearsonr(merged[col], merged[titer_col])
            corr_matrix.loc[col, day] = round(r, 3)

    corr_matrix = corr_matrix.astype(float)
    corr_matrix.index = [c.replace("_conc", "") for c in corr_matrix.index]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8},
                vmin=-1, vmax=1)
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Component", fontsize=11)
    ax.set_title("Day × Component correlation with final Titer (Pearson r)",
                 fontsize=12, pad=12)
    plt.tight_layout()
    out = RESULT_DIR / "ts_correlation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return str(out)


def fault_detection(filepath: str = None) -> str:
    """
    Fault flag 발생 배치 시각화.
    정상 배치 vs 이상 배치의 주요 성분 비교.
    ★ 신규
    """
    path = Path(filepath) if filepath else config.DATA_TIMESERIES
    df   = pd.read_csv(path)

    if "Fault flag" not in df.columns:
        raise ValueError("'Fault flag' column not found.")

    batch_col  = "Batch_ID"
    time_col   = "Time (day)"
    fault_col  = "Fault flag"
    plot_cols  = ["Glucose_conc", "Glutamine_conc", "Lactate_conc"]
    plot_cols  = [c for c in plot_cols if c in df.columns]

    # 배치별 fault 여부
    fault_batches  = set(df[df[fault_col] == 1][batch_col].unique())
    normal_batches = set(df[df[fault_col] == 0][batch_col].unique()) - fault_batches

    n    = len(plot_cols)
    fig, axes = plt.subplots(1, n, figsize=(n * 4.5, 4))
    if n == 1:
        axes = [axes]

    for i, col in enumerate(plot_cols):
        for bid in normal_batches:
            grp = df[df[batch_col] == bid].sort_values(time_col)
            axes[i].plot(grp[time_col], grp[col],
                         color="#1D9E75", alpha=0.15, lw=0.8)
        for bid in fault_batches:
            grp = df[df[batch_col] == bid].sort_values(time_col)
            axes[i].plot(grp[time_col], grp[col],
                         color="#E24B4A", alpha=0.6, lw=1.5)

        from matplotlib.lines import Line2D
        axes[i].legend(handles=[
            Line2D([0], [0], color="#1D9E75", lw=2, label=f"Normal ({len(normal_batches)})"),
            Line2D([0], [0], color="#E24B4A", lw=2, label=f"Fault ({len(fault_batches)})"),
        ], fontsize=8)
        axes[i].set_title(col.replace("_conc", ""), fontsize=11)
        axes[i].set_xlabel("Day")
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(f"Fault detection — {len(fault_batches)} fault batches detected",
                 fontsize=13)
    plt.tight_layout()
    out = RESULT_DIR / "fault_detection.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return str(out)