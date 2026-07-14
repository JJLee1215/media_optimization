"""
Models/xgboost_model.py
XGBoost Regression

Input:
  X_train : (n_train, 9)   initial media composition (scaled)
  X_test  : (n_test,  9)

  Features (9):
    Glucose_0, Glutamine_0, Asparagine_0
    Lactate_0, Ammonia_0
    Cu_0, Zn_0, Mn_0, Fe_0

  ※ Heterogeneity pipeline (use_pipeline=True) 적용 시:
     X_train/X_test는 (n, VECTOR_DIM)으로 변환된 상태로 들어옴
     (SMILES → embedding(RDKit/ChemBERTa/UniMol) → concat → pooling → scaler → (PCA))
     XGBoost 자체는 입력 차원에 무관하게 동작

Target:
  y_train : (n_train,)   titer_final
  y_test  : (n_test,)

Source: batch_table_syn.csv  via data_preprocess.get_static_data()

Methods:
  train()              fit model
  predict()             return predictions
  evaluate()             compute RMSE, R² + save plots
  feature_importance()  save feature importance plot
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
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

MODEL_NAME = "xgboost"
SAVE_PATH  = config.model_save_path(MODEL_NAME)
RESULT_DIR = config.result_dir(MODEL_NAME)


class XGBoostModel:
    def __init__(self):
        self.model  = XGBRegressor(
            n_estimators=config.XGB_N_ESTIMATORS,
            random_state=config.RANDOM_SEED,
            verbosity=0,
            n_jobs=-1,
        )
        self.scaler = None
        self.x_cols = None
        # ── heterogeneity pipeline 관련 (기본값) ──
        self.use_pipeline    = False
        self.embedding_model = None
        self.pipeline_obj     = None

    def train(self, X_train, y_train, x_cols=None, scaler=None):
        """
        X_train : (n_train, 9) or (n_train, VECTOR_DIM)  already scaled
        y_train : (n_train,)
        x_cols  : feature names (for importance plot)
        scaler  : fitted StandardScaler (saved for inference)
        """
        self.scaler = scaler
        self.x_cols = x_cols
        print(f"[XGBoost] Training...  n_train={len(X_train)}  n_features={X_train.shape[1]}")
        self.model.fit(X_train, y_train)
        print("[XGBoost] Training complete.")

    def predict(self, X):
        """Returns (y_pred, None) — None for API consistency with GP."""
        return self.model.predict(X), None

    def evaluate(self, X_test, y_test):
        """Compute RMSE, R² and save evaluation plots."""
        y_pred, _ = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        print(f"[XGBoost] RMSE: {rmse:.4f}  R²: {r2:.4f}")

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].scatter(y_test, y_pred, alpha=0.6, s=30, color="#1D9E75")
        lims = [min(y_test.min(), y_pred.min()) - 0.2,
                max(y_test.max(), y_pred.max()) + 0.2]
        axes[0].plot(lims, lims, "r--", lw=1)
        axes[0].set_xlabel("Actual")
        axes[0].set_ylabel("Predicted")
        axes[0].set_title(f"XGBoost  (R²={r2:.3f}  RMSE={rmse:.3f})")
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(y_pred - y_test, bins=15, edgecolor="white", color="#534AB7")
        axes[1].axvline(0, color="red", lw=1, ls="--")
        axes[1].set_title("Residuals")
        axes[1].set_xlabel("Predicted - Actual")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(RESULT_DIR / "eval.png", dpi=120)
        plt.close()
        print(f"[XGBoost] Plot saved: {RESULT_DIR}/eval.png")

        return {
            "model": MODEL_NAME,
            "rmse" : round(float(rmse), 4),
            "r2"   : round(float(r2),   4),
        }

    def feature_importance(self, x_cols=None):
        cols = x_cols or self.x_cols
        if cols is None:
            print("[XGBoost] x_cols not provided, skipping feature importance.")
            return

        imp = self.model.feature_importances_

        if len(cols) != len(imp):
            cols = [f"feature_{i}" for i in range(len(imp))]

        idx = np.argsort(imp)

        if len(idx) > 20:
            idx = idx[-20:]

        fig, ax = plt.subplots(figsize=(6, 8))
        ax.barh(range(len(idx)), imp[idx], color="#1D9E75", edgecolor="white")
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([cols[i] for i in idx], fontsize=8)
        ax.set_title("Feature Importance (Top 20)")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "feature_importance.png", dpi=120)
        plt.close()
        print(f"[XGBoost] Feature importance saved: {RESULT_DIR}/feature_importance.png")

    def get_config(self):
        """
        학습 정보 기록용 하이퍼파라미터 리포트.
        train.py의 train_model()이 result.json의 meta.hyperparams에
        이 값을 그대로 저장함.
        """
        return {
            "n_estimators": self.model.n_estimators,
            "random_state": self.model.random_state,
        }

    def cross_validate(self, X, y, cv=5):
        """k-fold cross validation."""
        scores = cross_val_score(
            XGBRegressor(
                n_estimators=config.XGB_N_ESTIMATORS,
                random_state=config.RANDOM_SEED,
                verbosity=0, n_jobs=-1,
            ),
            X, y, cv=cv, scoring="r2", n_jobs=-1
        )
        print(f"[XGBoost] CV R²: {np.round(scores, 3)}"
              f"  mean={scores.mean():.3f} ± {scores.std():.3f}")
        return scores

    def save(self, use_pipeline: bool = False, embedding_model: str = None, pipeline_obj=None):
        """
        모델을 pkl로 저장.

        use_pipeline    : 학습 시 heterogeneity pipeline 사용 여부.
        embedding_model : "rdkit" | "chemberta" | "unimol" | None
        pipeline_obj    : 실제 학습에 쓰인 파이프라인 인스턴스
                          (내부에 fit된 scaler + PCA가 이미 들어있음).
                          predict.py는 load() 후 .transform()만 호출할 것
                          — .fit_transform() 호출 금지.
                          use_pipeline=False면 None.
        """
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
        print(f"[XGBoost] Saved: {SAVE_PATH}  "
              f"(use_pipeline={self.use_pipeline}, embedding_model={self.embedding_model})")

    def load(self):
        with open(SAVE_PATH, "rb") as f:
            data = pickle.load(f)
        self.model  = data["model"]
        self.scaler = data["scaler"]
        self.x_cols = data["x_cols"]
        self.use_pipeline = data.get("use_pipeline", False)
        self.embedding_model = data.get("embedding_model", None)
        self.pipeline_obj     = data.get("pipeline_obj", None)
        print(f"[XGBoost] Loaded: {SAVE_PATH}  "
              f"(use_pipeline={self.use_pipeline}, embedding_model={self.embedding_model})")


if __name__ == "__main__":
    from data_preprocess import get_static_data
    X_train, X_test, y_train, y_test, x_cols, scaler, emb, dim = get_static_data()

    model = XGBoostModel()
    model.train(X_train, y_train, x_cols=x_cols, scaler=scaler)
    model.evaluate(X_test, y_test)
    model.feature_importance()
    model.cross_validate(X_train, y_train)
    model.save()