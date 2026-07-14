"""
Models/gaussian_process.py
Gaussian Process Regression

Input:
  X_train : (n_train, 9) or (n_train, VECTOR_DIM)   initial media composition (scaled)
  X_test  : (n_test,  9) or (n_test,  VECTOR_DIM)

  Features (9):
    Glucose_0, Glutamine_0, Asparagine_0
    Lactate_0, Ammonia_0
    Cu_0, Zn_0, Mn_0, Fe_0

  ※ Heterogeneity pipeline (use_pipeline=True) 적용 시:
     X_train/X_test는 (n, VECTOR_DIM)으로 변환된 상태로 들어옴
     (SMILES → embedding(RDKit/ChemBERTa/UniMol) → concat → pooling → scaler → (PCA))
     GaussianProcess 자체는 입력 차원에 무관하게 동작

Target:
  y_train : (n_train,)   titer_final
  y_test  : (n_test,)

Source: batch_table_syn.csv  via data_preprocess.get_static_data()

Methods:
  train()              fit model
  predict()             return predictions
  evaluate()             compute RMSE, R² + save plots
  cross_validate()       k-fold cross validation
  save() / load()        persist model to disk

  ※ save(use_pipeline=..., embedding_model=..., pipeline_obj=...) :
     학습 시 heterogeneity pipeline 사용 여부, 파이프라인 종류,
     그리고 실제 fit된 파이프라인 인스턴스(scaler+PCA 포함)를 pkl에
     함께 기록. predict.py가 이 값들을 읽어서 예측 시에도 동일한
     전처리(pipeline.transform())를 자동으로 재현할 수 있게 함.
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

MODEL_NAME = "gaussian_process"
SAVE_PATH  = config.model_save_path(MODEL_NAME)
RESULT_DIR = config.result_dir(MODEL_NAME)


class GaussianProcessModel:
    def __init__(self):
        self.model  = self._build()
        self.scaler = None
        self.x_cols = None
        self.use_pipeline    = False
        self.embedding_model = None
        self.pipeline_obj     = None

    def _build(self):
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        return GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=config.GP_N_RESTARTS,
            normalize_y=True,
            random_state=config.RANDOM_SEED,
        )

    def train(self, X_train, y_train, x_cols=None, scaler=None):
        self.scaler = scaler
        self.x_cols = x_cols
        print(f"[GaussianProcess] Training...  n_train={len(X_train)}  n_features={X_train.shape[1]}")
        self.model.fit(X_train, y_train)
        print(f"[GaussianProcess] Training complete.")
        print(f"[GaussianProcess] Learned kernel: {self.model.kernel_}")

    def predict(self, X):
        y_pred, y_std = self.model.predict(X, return_std=True)
        return y_pred, y_std

    def evaluate(self, X_test, y_test):
        y_pred, y_std = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        print(f"[GaussianProcess] RMSE: {rmse:.4f}  R2: {r2:.4f}")

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.errorbar(y_test, y_pred, yerr=y_std, fmt="o", alpha=0.6,
                    ecolor="#5DCAA5", color="#0F6E56", markersize=6, capsize=3)
        lims = [min(y_test.min(), y_pred.min()) - 0.2,
                max(y_test.max(), y_pred.max()) + 0.2]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Gaussian Process  (R2={r2:.3f}  RMSE={rmse:.3f})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "eval.png", dpi=120)
        plt.close()
        print(f"[GaussianProcess] Plot saved: {RESULT_DIR}/eval.png")

        return {
            "model": MODEL_NAME,
            "rmse" : round(float(rmse), 4),
            "r2"   : round(float(r2),   4),
        }

    def get_config(self):
        return {
            "kernel"               : str(self.model.kernel_) if hasattr(self.model, "kernel_") else None,
            "n_restarts_optimizer" : config.GP_N_RESTARTS,
            "normalize_y"          : True,
            "random_state"         : config.RANDOM_SEED,
        }

    def cross_validate(self, X, y, cv=5):
        scores = cross_val_score(
            self._build(), X, y,
            cv=cv, scoring="r2", n_jobs=-1
        )
        print(f"[GaussianProcess] CV R2: {np.round(scores, 3)}"
              f"  mean={scores.mean():.3f} +/- {scores.std():.3f}")
        return scores

    def save(self, use_pipeline=False, embedding_model=None, pipeline_obj=None):
        self.use_pipeline    = use_pipeline
        self.embedding_model = embedding_model
        self.pipeline_obj     = pipeline_obj

        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SAVE_PATH, "wb") as f:
            pickle.dump({
                "model"           : self.model,
                "scaler"          : self.scaler,
                "x_cols"          : self.x_cols,
                "use_pipeline"    : self.use_pipeline,
                "embedding_model" : self.embedding_model,
                "pipeline_obj"    : self.pipeline_obj,
            }, f)
        print(f"[GaussianProcess] Saved: {SAVE_PATH}  "
              f"(use_pipeline={self.use_pipeline}, embedding_model={self.embedding_model})")

    def load(self):
        with open(SAVE_PATH, "rb") as f:
            data = pickle.load(f)
        self.model  = data["model"]
        self.scaler = data["scaler"]
        self.x_cols = data["x_cols"]
        self.use_pipeline    = data.get("use_pipeline", False)
        self.embedding_model = data.get("embedding_model", None)
        self.pipeline_obj     = data.get("pipeline_obj", None)
        print(f"[GaussianProcess] Loaded: {SAVE_PATH}  "
              f"(use_pipeline={self.use_pipeline}, embedding_model={self.embedding_model})")


if __name__ == "__main__":
    from data_preprocess import get_static_data
    X_train, X_test, y_train, y_test, x_cols, scaler, emb, dim = get_static_data()

    model = GaussianProcessModel()
    model.train(X_train, y_train, x_cols=x_cols, scaler=scaler)
    model.evaluate(X_test, y_test)
    model.cross_validate(X_train, y_train)
    model.save()
