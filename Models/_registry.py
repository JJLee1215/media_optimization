"""
Models/_registry.py
Model registry — single source of truth for all available models.

To add a new model:
  1. Add model code to Models/
  2. Add entry here
  3. UI will automatically reflect the change
"""

MODEL_REGISTRY = [
    # ── Static models ─────────────────────────────────
    {
        "id"     : "gaussian_process",
        "name"   : "GP",
        "desc"   : "Gaussian Process",
        "icon"   : "📈",
        "section": "static",
    },
    {
        "id"     : "xgboost",
        "name"   : "XGBoost",
        "desc"   : "Gradient Boosting",
        "icon"   : "⚡",
        "section": "static",
    },
    {
        "id"     : "random_forest",
        "name"   : "RF",
        "desc"   : "Random Forest",
        "icon"   : "🌲",
        "section": "static",
    },
    {
        "id"     : "mlp",
        "name"   : "MLP",
        "desc"   : "Neural Network",
        "icon"   : "🧠",
        "section": "static",
    },

    # ── Time series models ────────────────────────────
    {
        "id"     : "rnn",
        "name"   : "RNN",
        "desc"   : "Recurrent Network",
        "icon"   : "🔁",
        "section": "timeseries",
    },
    {
        "id"     : "lstm",
        "name"   : "LSTM",
        "desc"   : "Long Short-Term",
        "icon"   : "⏱️",
        "section": "timeseries",
    },
    {
        "id"     : "transformer",
        "name"   : "Transformer",
        "desc"   : "Attention-based",
        "icon"   : "⚙️",
        "section": "timeseries",
    },
    # ── TCN: LSTM/RNN보다 학습 빠르고 안정적 ─────────
    {
        "id"     : "tcn",
        "name"   : "TCN",
        "desc"   : "Temporal Conv Network",
        "icon"   : "🌊",
        "section": "timeseries",
    },

    # ── Static & Time Series models ───────────────────
    {
        "id"     : "static_time_gnn",
        "name"   : "StaticTimeGNN",
        "desc"   : "Static + Timeseries",
        "icon"   : "🕸️",
        "section": "static_time",
    },
]


def get_registry_by_section():
    """Return models grouped by section."""
    result = {"static": [], "timeseries": [], "static_time": []}
    for m in MODEL_REGISTRY:
        result[m["section"]].append(m)
    return result