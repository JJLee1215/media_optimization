"""
app.py
FastAPI 백엔드

Routes:
  POST /train          데이터셋 + 모델 선택 → 학습 + 결과 반환
  GET  /train/status   학습 진행 상태
  POST /predict        배지 조성 + 모델 선택 → titer 예측
"""

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
import pickle, os, json
from pathlib import Path

app = FastAPI(title="Bioprocess Engineering Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 경로 ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("data_file")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

X_COLS = [
    "Aeration rate",
    "Agitator RPM",
    "Sugar feed rate",
    "Acid flow rate",
    "Base flow rate",
    "Heating/cooling water flow rate",
    "Heating water flow rate",
    "Water for injection/dilution",
    "PAA flow",
    "Oil flow",
]

# ── 학습 상태 저장 (간단한 in-memory) ─────────────────────────────────────────
train_status = {"status": "idle", "message": "", "result": None}


# ── 스키마 ────────────────────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    dataset:  str = "batch_table.csv"   # data_file/ 아래 파일명
    model:    str = "gp"                # gp | xgboost | random_forest | mlp
    test_size: float = 0.2

class PredictRequest(BaseModel):
    model: str = "gp"
    components: dict                    # {"Aeration rate": 30.0, ...}


# ── 모델 빌드 ─────────────────────────────────────────────────────────────────
def build_model(name: str):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor

    if name == "gp":
        kernel = (
            ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1.0)
        )
        return GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=3,
            normalize_y=True, random_state=42
        )
    elif name == "xgboost":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif name == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(32, 16), max_iter=300,
            random_state=42, early_stopping=True
        )
    else:
        raise ValueError(f"Unknown model: {name}")


# ── 학습 함수 (백그라운드) ────────────────────────────────────────────────────
def run_training(dataset: str, model_name: str, test_size: float):
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score

    train_status["status"]  = "running"
    train_status["message"] = "데이터 로딩 중..."

    try:
        # 데이터 로드
        path = DATA_DIR / dataset
        df   = pd.read_csv(path)
        x_cols = [c for c in X_COLS if c in df.columns]
        X = df[x_cols].values.astype(np.float64)
        y = df["titer_final"].values.astype(np.float64)

        train_status["message"] = "모델 학습 중..."

        # 분리 + 스케일링
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        scaler  = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        X_s       = scaler.transform(X)

        # 학습
        model = build_model(model_name)
        model.fit(X_train_s, y_train)

        # 평가
        y_pred = model.predict(X_test_s)
        rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2     = float(r2_score(y_test, y_pred))

        # CV
        cv = cross_val_score(build_model(model_name), X_s, y, cv=5, scoring="r2")

        # 모델 저장
        save = {"model": model, "scaler": scaler, "x_cols": x_cols}
        with open(MODEL_DIR / f"{model_name}.pkl", "wb") as f:
            pickle.dump(save, f)

        train_status["status"]  = "done"
        train_status["message"] = "학습 완료"
        train_status["result"]  = {
            "model":       model_name,
            "n_train":     int(len(X_train)),
            "n_test":      int(len(X_test)),
            "rmse":        round(rmse, 4),
            "r2":          round(r2, 4),
            "cv_r2_mean":  round(float(cv.mean()), 4),
            "cv_r2_std":   round(float(cv.std()), 4),
            "y_test":      y_test.tolist(),
            "y_pred":      y_pred.tolist(),
        }

    except Exception as e:
        train_status["status"]  = "error"
        train_status["message"] = str(e)


# ── 라우트 ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Bioprocess Engineering Tool API"}


@app.get("/datasets")
def list_datasets():
    files = [f.name for f in DATA_DIR.glob("*.csv")]
    return {"datasets": files}


@app.post("/train")
def train(req: TrainRequest, bg: BackgroundTasks):
    train_status["status"]  = "pending"
    train_status["message"] = "학습 대기 중..."
    train_status["result"]  = None
    bg.add_task(run_training, req.dataset, req.model, req.test_size)
    return {"message": "학습 시작됨"}


@app.get("/train/status")
def get_train_status():
    return train_status


@app.post("/predict")
def predict(req: PredictRequest):
    model_path = MODEL_DIR / f"{req.model}.pkl"
    if not model_path.exists():
        return {"error": f"{req.model} 모델이 없습니다. 먼저 학습해주세요."}

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    model   = saved["model"]
    scaler  = saved["scaler"]
    x_cols  = saved["x_cols"]

    # 입력값 조립
    x_vals = [req.components.get(c, 0.0) for c in x_cols]
    X_input = scaler.transform(np.array(x_vals).reshape(1, -1))

    # 예측
    if req.model == "gp":
        mu, sigma = model.predict(X_input, return_std=True)
        return {
            "titer_pred": round(float(mu[0]), 4),
            "sigma":      round(float(sigma[0]), 4),
            "model":      req.model,
        }
    else:
        y_pred = model.predict(X_input)
        return {
            "titer_pred": round(float(y_pred[0]), 4),
            "sigma":      None,
            "model":      req.model,
        }