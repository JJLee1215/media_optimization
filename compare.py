"""
compare.py
Model comparison utilities

Functions:
  collect_results()   load result JSONs from all models
  print_table()       print comparison table to terminal
  plot_comparison()   save RMSE bar chart
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import config

COMPARISON_DIR = config.RESULTS_TT_DIR / "comparison"

STATIC_MODELS      = ["gaussian_process", "random_forest", "xgboost", "mlp"]
TIME_MODELS        = ["rnn", "lstm", "transformer"]
STATIC_TIME_MODELS = ["static_time_gnn"]
ALL_MODELS         = STATIC_MODELS + TIME_MODELS + STATIC_TIME_MODELS


def collect_results(mode: str = "train") -> dict:
    """
    Load result JSONs from all trained/tested models.

    Args:
        mode: "train" or "test"

    Returns:
        results dict  {model_name: {rmse, r2, ...}}
    """
    filename = f"{mode}_result.json"
    results  = {}

    for model_name in ALL_MODELS:
        path = config.result_dir(model_name) / filename
        if path.exists():
            with open(path) as f:
                results[model_name] = json.load(f)
        else:
            results[model_name] = None

    return results


def print_table(results: dict, mode: str = "train"):
    """Print comparison table to terminal."""
    print(f"\n{'='*60}")
    print(f"  Model Comparison — {mode.upper()}")
    print(f"{'='*60}")
    print(f"  {'Model':<22} {'RMSE':>8}  {'R²':>8}  {'Type':<12}")
    print(f"  {'-'*55}")

    groups = {
        "static"      : STATIC_MODELS,
        "time"        : TIME_MODELS,
        "static_time" : STATIC_TIME_MODELS,
    }

    for group, models in groups.items():
        for name in models:
            r = results.get(name)
            if r is None:
                print(f"  {name:<22}  {'—':>8}  {'—':>8}  {group}")
                continue

            rmse = r.get("rmse") or r.get("titer_rmse", "—")
            r2   = r.get("r2", "—")

            rmse_str = f"{rmse:.4f}" if isinstance(rmse, float) else rmse
            r2_str   = f"{r2:.4f}"   if isinstance(r2,   float) else r2
            print(f"  {name:<22} {rmse_str:>8}  {r2_str:>8}  {group}")

        print(f"  {'-'*55}")

    print()


def plot_comparison(results: dict, mode: str = "train"):
    """Save RMSE bar chart."""
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    names, rmses, colors = [], [], []
    color_map = {
        "gaussian_process" : "#1D9E75",
        "random_forest"    : "#0F6E56",
        "xgboost"          : "#534AB7",
        "mlp"              : "#7F77DD",
        "rnn"              : "#BA7517",
        "lstm"             : "#EF9F27",
        "transformer"      : "#FAC775",
        "static_time_gnn"  : "#E24B4A",
    }

    for name in ALL_MODELS:
        r = results.get(name)
        if r is None:
            continue
        rmse = r.get("rmse") or r.get("titer_rmse")
        if rmse is None:
            continue
        names.append(name.replace("_", "\n"))
        rmses.append(rmse)
        colors.append(color_map.get(name, "#888780"))

    if not names:
        print("[compare] No results to plot.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    bars = ax.bar(range(len(names)), rmses, color=colors, edgecolor="white", width=0.6)

    # Value labels on bars
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title(f"Model Comparison — RMSE ({mode.upper()})",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(rmses) * 1.2)

    # Group separators
    n_static = len([n for n in STATIC_MODELS if n.replace("_", "\n") in names])
    n_time   = len([n for n in TIME_MODELS   if n.replace("_", "\n") in names])
    if n_static > 0 and n_time > 0:
        ax.axvline(n_static - 0.5, color="#888780", lw=0.8, ls="--", alpha=0.5)
    if n_time > 0 and len(STATIC_TIME_MODELS) > 0:
        ax.axvline(n_static + n_time - 0.5, color="gray", lw=0.8, ls="--", alpha=0.5)

    plt.tight_layout()
    path = COMPARISON_DIR / f"comparison_rmse_{mode}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[compare] Chart saved: {path}")

    # Save JSON
    summary = {name.replace("\n", "_"): rmse
               for name, rmse in zip(names, rmses)}
    json_path = COMPARISON_DIR / f"comparison_{mode}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[compare] JSON  saved: {json_path}")


if __name__ == "__main__":
    results = collect_results("train")
    print_table(results, "train")
    plot_comparison(results, "train")