"""
routers/predict.py
Prediction routes

  POST /predict   input media composition → predict titer / viability

Supported models:
  gaussian_process   → titer_pred + sigma
  random_forest      → titer_pred
  xgboost            → titer_pred
  mlp                → titer_pred
  static_time_gnn    → titer_pred + viab_pred
"""

import pickle
import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config

router = APIRouter(prefix="/predict", tags=["Prediction"])

STATIC_MODELS = ["gaussian_process", "random_forest", "xgboost", "mlp"]


class PredictRequest(BaseModel):
    model : str    # model name
    inputs: dict   # {"Glucose_0": 4.5, "Glutamine_0": 2.0, ...}


@router.post("")
def predict(req: PredictRequest):
    """
    Predict titer (and viability for StaticTimeGNN).

    Static models:
      inputs = initial media composition
      returns titer_pred (+ sigma for GP)

    StaticTimeGNN:
      inputs = initial media composition
      returns titer_pred + viab_pred
    """
    model_name = req.model
    save_path  = config.model_save_path(model_name)

    if not save_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No trained model: {model_name}. Run training first."
        )

    # ── Static models ──────────────────────────────
    if model_name in STATIC_MODELS:
        with open(save_path, "rb") as f:
            saved = pickle.load(f)

        model  = saved["model"]
        scaler = saved["scaler"]
        x_cols = saved["x_cols"]

        x_vals  = np.array([req.inputs.get(c, 0.0) for c in x_cols]).reshape(1, -1)
        X_input = scaler.transform(x_vals)

        if model_name == "gaussian_process":
            mu, sigma = model.predict(X_input, return_std=True)
            return {
                "model"      : model_name,
                "titer_pred" : round(float(mu[0]), 4),
                "sigma"      : round(float(sigma[0]), 4),
                "viab_pred"  : None,
            }
        else:
            y_pred = model.predict(X_input)
            return {
                "model"      : model_name,
                "titer_pred" : round(float(y_pred[0]), 4),
                "sigma"      : None,
                "viab_pred"  : None,
            }

    # ── StaticTimeGNN ──────────────────────────────
    elif model_name == "static_time_gnn":
        import torch
        from Models.StaticTimeGNN import StaticTimeGNNModel

        df_static  = pd.read_csv(config.DATA_STATIC)
        df_dynamic = pd.read_csv(config.DATA_TIMESERIES)

        drop_cols   = ["batch_id", "titer_final", "viab_final"]
        static_cols = [c for c in df_static.columns if c not in drop_cols]

        batch_col  = "Batch ID"
        time_col   = "Time (day)"
        target_col = "Titer (g/L)"
        skip_cols  = [batch_col, time_col, "Fault flag", target_col]
        feat_cols  = [c for c in df_dynamic.columns if c not in skip_cols]

        N  = len(feat_cols)
        A0 = torch.zeros(N, N)

        gnn = StaticTimeGNNModel(
            d_static  = len(static_cols),
            d_dynamic = N,
            N         = N,
            A0        = A0,
        )
        gnn.load()

        # m_static from inputs
        m_vals = [req.inputs.get(c, 0.0) for c in static_cols]
        m      = torch.tensor([m_vals], dtype=torch.float32)

        # X_dynamic: use mean of all batches as proxy
        X_list = []
        for _, grp in df_dynamic.groupby(batch_col):
            grp = grp.sort_values(time_col)
            X_list.append(grp[feat_cols].values)
        T     = min(len(x) for x in X_list)
        X_mean = np.mean([x[:T] for x in X_list], axis=0)
        X      = torch.tensor([X_mean], dtype=torch.float32)

        mu, viab, _ = gnn.predict(m, X)

        return {
            "model"      : model_name,
            "titer_pred" : round(float(mu[0]), 4),
            "sigma"      : None,
            "viab_pred"  : round(float(viab[0]), 4),
        }

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")