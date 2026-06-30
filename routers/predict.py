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

※ Heterogeneity pipeline (use_pipeline) 처리:
  Static models(gaussian_process/random_forest/xgboost/mlp)는
  학습 시 pipeline ON으로 학습되면 scaler가 230차원 기준으로 fit됨.
  pkl/pt 저장 시 use_pipeline 플래그가 함께 기록되므로,
  예측 시에도 saved["use_pipeline"]을 읽어서 동일하게
  9개 raw 입력값 → MediaPipeline.transform() → 230차원으로 변환 후
  scaler.transform()에 넣어야 함.
  (이 처리가 없으면 ValueError: X has 9 features, but StandardScaler
   is expecting 230 features 에러 발생)

  StaticTimeGNN은 현재 save()가 state_dict만 저장하는 구조라
  use_pipeline 메타정보를 기록하지 못함 → 추후 개선 예정 (보류 중).
"""

import pickle
import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config

router = APIRouter(prefix="/predict", tags=["Prediction"])

STATIC_MODELS = ["gaussian_process", "random_forest", "xgboost", "mlp"]

# ── pipeline 인스턴스 캐시 (요청마다 새로 만들지 않도록) ──
# MediaPipeline 초기화 시 RDKit descriptor 계산기를 만드는데,
# 이게 가벼운 작업은 아니라서 모듈 레벨에서 한 번만 생성해 재사용함.
_pipeline_cache = None


def get_pipeline():
    global _pipeline_cache
    if _pipeline_cache is None:
        from heterogeneity.smile_gem_pipe import MediaPipeline
        _pipeline_cache = MediaPipeline()
    return _pipeline_cache


class PredictRequest(BaseModel):
    model : str    # model name
    inputs: dict   # {"Glucose_0": 4.5, "Glutamine_0": 2.0, ...}


@router.post("")
def predict(req: PredictRequest):
    """
    Predict titer (and viability for StaticTimeGNN).

    Static models:
      inputs = initial media composition (raw, 9개 컴포넌트)
      saved["use_pipeline"]에 따라 자동으로 pipeline 적용 여부 결정
      returns titer_pred (+ sigma for GP)

    StaticTimeGNN:
      inputs = initial media composition
      returns titer_pred + viab_pred
      ※ pipeline 적용 여부 자동 판단 미지원 (보류 중)
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
        # ── 구버전 pkl 호환: use_pipeline 키가 없으면 False로 간주 ──
        use_pipeline = saved.get("use_pipeline", False)

        # x_cols는 항상 raw 9개 컴포넌트 이름 (pipeline 적용 여부와 무관)
        x_vals = np.array([req.inputs.get(c, 0.0) for c in x_cols]).reshape(1, -1)

        if use_pipeline:
            # ── 학습 때와 동일하게 9 → 230차원 변환 ──
            pipeline = get_pipeline()
            x_vals   = pipeline.transform(x_vals, x_cols)

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

        # ※ 현재 StaticTimeGNN은 use_pipeline 메타정보를 저장하지 않으므로
        #   항상 d_static=len(static_cols)(=9)로 고정. pipeline 적용 학습은
        #   아직 지원하지 않음 (보류 중).
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