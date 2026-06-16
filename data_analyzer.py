"""
data_analyzer.py
CSV file analysis tool

1. Read CSV file
2. Analyze dimensions
3. Analyze features (stats, types, missing values)
4. Generate and save plots
5. (Future: UI integration — file select → show dims, features, plots)

Usage:
  python data_analyzer.py --file data_file/batch_table_syn.csv
  python data_analyzer.py --file data_file/timeseries_syn.csv
  python data_analyzer.py --file data_file/batch_table_syn.csv --out outputs/analysis
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ══════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════

DEFAULT_OUT_DIR = Path("outputs/analysis")
PLOT_DPI        = 120
PLOT_FIGSIZE_SM = (8, 5)
PLOT_FIGSIZE_LG = (14, 8)


# ══════════════════════════════════════════════
# 1. Read CSV
# ══════════════════════════════════════════════

def read_csv(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    print(f"\n[File] {path}")
    return df


# ══════════════════════════════════════════════
# 2. Analyze dimensions
# ══════════════════════════════════════════════

def analyze_dimensions(df: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape

    info = {
        "n_rows"   : n_rows,
        "n_cols"   : n_cols,
        "columns"  : list(df.columns),
        "dtypes"   : {col: str(df[col].dtype) for col in df.columns},
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 4),
    }

    print(f"\n{'='*55}")
    print(f"  Dimensions")
    print(f"{'='*55}")
    print(f"  Rows      : {n_rows:,}")
    print(f"  Columns   : {n_cols}")
    print(f"  Memory    : {info['memory_mb']} MB")
    print(f"\n  Columns ({n_cols}):")
    for col in df.columns:
        print(f"    {col:<40} {str(df[col].dtype)}")

    return info


# ══════════════════════════════════════════════
# 3. Analyze features
# ══════════════════════════════════════════════

def analyze_features(df: pd.DataFrame) -> dict:
    numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    missing          = df.isnull().sum()
    missing_cols     = missing[missing > 0].to_dict()

    stats = df[numeric_cols].describe().round(4).to_dict()

    print(f"\n{'='*55}")
    print(f"  Feature Analysis")
    print(f"{'='*55}")
    print(f"  Numeric columns    : {len(numeric_cols)}")
    print(f"  Categorical columns: {len(categorical_cols)}")

    if missing_cols:
        print(f"\n  Missing values:")
        for col, cnt in missing_cols.items():
            print(f"    {col:<40} {cnt} ({cnt/len(df)*100:.1f}%)")
    else:
        print(f"\n  Missing values: None")

    print(f"\n  Numeric feature stats:")
    print(f"  {'Column':<35} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*67}")
    for col in numeric_cols:
        print(f"  {col:<35} "
              f"{df[col].min():>8.3f} "
              f"{df[col].max():>8.3f} "
              f"{df[col].mean():>8.3f} "
              f"{df[col].std():>8.3f}")

    return {
        "numeric_cols"    : numeric_cols,
        "categorical_cols": categorical_cols,
        "missing"         : missing_cols,
        "stats"           : stats,
    }


# ══════════════════════════════════════════════
# 4. Generate plots
# ══════════════════════════════════════════════

def generate_plots(df: pd.DataFrame, out_dir: Path, file_stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    saved = []

    # ── Plot 1: Distribution of each numeric feature ──
    n = len(numeric_cols)
    if n > 0:
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3.5, nrows * 3))
        axes = np.array(axes).flatten()

        for i, col in enumerate(numeric_cols):
            axes[i].hist(df[col].dropna(), bins=20,
                         color="#1D9E75", edgecolor="white", alpha=0.8)
            axes[i].set_title(col, fontsize=9)
            axes[i].set_xlabel("Value", fontsize=8)
            axes[i].set_ylabel("Count", fontsize=8)
            axes[i].tick_params(labelsize=7)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f"Feature Distributions — {file_stem}",
                     fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        path = out_dir / f"{file_stem}_distributions.png"
        plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        saved.append(path)
        print(f"  Saved: {path}")

    # ── Plot 2: Correlation heatmap ──
    if n > 1:
        corr = df[numeric_cols].corr()
        fig_h = max(6, n * 0.5)
        fig, ax = plt.subplots(figsize=(fig_h + 2, fig_h))
        sns.heatmap(corr, ax=ax, cmap="RdYlGn", center=0,
                    vmin=-1, vmax=1, annot=(n <= 15),
                    fmt=".2f", linewidths=0.4,
                    cbar_kws={"label": "Correlation"})
        ax.set_title(f"Correlation Heatmap — {file_stem}",
                     fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)
        plt.tight_layout()
        path = out_dir / f"{file_stem}_correlation.png"
        plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        saved.append(path)
        print(f"  Saved: {path}")

    # ── Plot 3: Time series plot (if time column detected) ──
    time_col = next(
        (c for c in df.columns if "time" in c.lower() or "day" in c.lower()), None
    )
    batch_col = next(
        (c for c in df.columns if "batch" in c.lower()), None
    )

    if time_col and batch_col:
        # Pick up to 5 random batches to plot
        batches = df[batch_col].unique()
        sample_batches = batches[:min(5, len(batches))]

        # Pick up to 6 numeric features (exclude batch/time/fault cols)
        skip = {batch_col.lower(), time_col.lower(), "fault flag", "fault_flag"}
        plot_cols = [c for c in numeric_cols
                     if c.lower() not in skip][:6]

        if plot_cols:
            ncols = min(3, len(plot_cols))
            nrows = (len(plot_cols) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(ncols * 4.5, nrows * 3.5))
            axes = np.array(axes).flatten()

            colors = ["#1D9E75", "#534AB7", "#E24B4A",
                      "#BA7517", "#0F6E56", "#7F77DD"]

            for i, col in enumerate(plot_cols):
                for j, bid in enumerate(sample_batches):
                    batch_df = df[df[batch_col] == bid].sort_values(time_col)
                    axes[i].plot(batch_df[time_col], batch_df[col],
                                 color=colors[j % len(colors)],
                                 alpha=0.7, linewidth=1.2,
                                 label=f"Batch {bid}")
                axes[i].set_title(col, fontsize=9)
                axes[i].set_xlabel(time_col, fontsize=8)
                axes[i].tick_params(labelsize=7)
                if i == 0:
                    axes[i].legend(fontsize=7)

            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            plt.suptitle(f"Time Series — {file_stem} (sample batches)",
                         fontsize=12, fontweight="bold", y=1.01)
            plt.tight_layout()
            path = out_dir / f"{file_stem}_timeseries.png"
            plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
            plt.close()
            saved.append(path)
            print(f"  Saved: {path}")

    return [str(p) for p in saved]


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def analyze(file_path: str, out_dir: str = None) -> dict:
    """
    Full analysis pipeline.
    Returns summary dict (for future UI integration).
    """
    out_path  = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    file_stem = Path(file_path).stem

    # 1. Read
    df = read_csv(file_path)

    # 2. Dimensions
    dim_info = analyze_dimensions(df)

    # 3. Features
    feat_info = analyze_features(df)

    # 4. Plots
    print(f"\n{'='*55}")
    print(f"  Generating plots → {out_path}/")
    print(f"{'='*55}")
    plots = generate_plots(df, out_path, file_stem)

    # 5. Save summary JSON
    summary = {
        "file"      : str(file_path),
        "dimensions": dim_info,
        "features"  : {
            "numeric_cols"    : feat_info["numeric_cols"],
            "categorical_cols": feat_info["categorical_cols"],
            "missing"         : feat_info["missing"],
        },
        "plots"     : plots,
    }

    json_path = out_path / f"{file_stem}_summary.json"
    out_path.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\n{'='*55}")
    print(f"  Analysis complete.")
    print(f"  Output: {out_path}/")
    print(f"{'='*55}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV Data Analyzer")
    parser.add_argument("--file", type=str, required=True,
                        help="Path to CSV file")
    parser.add_argument("--out",  type=str, default=None,
                        help="Output directory for plots and summary")
    args = parser.parse_args()

    analyze(args.file, args.out)