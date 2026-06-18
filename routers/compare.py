"""
routers/compare.py
Model Comparison routes

  GET /compare    return comparison results + chart URL
"""

from fastapi import APIRouter

import config

router = APIRouter(prefix="/compare", tags=["Compare"])


@router.get("")
def compare(mode: str = "train"):
    """Return comparison results + chart URL."""
    from compare import collect_results, plot_comparison

    results = collect_results(mode)
    plot_comparison(results, mode)

    chart_path = config.RESULTS_TT_DIR / "comparison" / f"comparison_rmse_{mode}.png"
    chart_url  = (
        f"/static/comparison/comparison_rmse_{mode}.png"
        if chart_path.exists() else None
    )

    return {
        "mode"     : mode,
        "results"  : results,
        "chart_url": chart_url,
    }