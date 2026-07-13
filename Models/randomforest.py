"""
Models/randomforest.py
Random Forest Regression

Input:
  X_train : (n_train, 9)   initial media composition (scaled)
  X_test  : (n_test,  9)

  Features (9):
    Glucose_0, Glutamine_0, Asparagine_0
    Lactate_0, Ammonia_0
    Cu_0, Zn_0, Mn_0, Fe_0

  ※ Heterogeneity pipeline (use_pipeline=True) 적용 시:
     X_train/X_test는 (n, 230)으로 변환된 상태로 들어옴
     (SMILES → RDKit descriptor → GEM 벡터 → Mean Pooling)
     RandomForest 자체는 입력 차원에 무관하게 동작

Target:
  y_train : (n_train,)   titer_final
  y_test  : (n_test,)

Source: batch_table_syn.csv  via preprocess.get_static_data()

Methods:
  train()              fit model
  predict()            return predictions
  evaluate()           compute RMSE, R² + save plots
  feature_importance() save feature importance plot
  cross_validate()     k-fold cross validation
  save() / load()      persist model to disk

  ※ save(use_pipeline=...) : 학습 시 heterogeneity pipeline 사용 여부를
     pkl 파일에 함께 기록. predict.py가 이 값을 읽어서
     예측 시에도 동일하게 pipeline을 적용할지 자동 판단.
     (학습 차원과 예측 차원이 다르면 StandardScaler.transform()에서
      ValueError: X has N features, but StandardScaler is expecting M features
      에러가 발생하므로, 저장된 값으로 분기 처리가 필요함)
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

MODEL_NAME = "random_forest"
SAVE_PATH  = config.model_save_path(MODEL_NAME)
RESULT_DIR = config.result_dir(MODEL_NAME)


class RandomForestModel:
    def __init__(self):
        self.model  = RandomForestRegressor(
            n_estimators=config.RF_N_ESTIMATORS,
            random_state=config.RANDOM_SEED,
            n_jobs=-1
        )
        self.scaler  = None
        self.x_cols  = None
        # ── heterogeneity pipeline 사용 여부 (기본 False) ──
        # train_static()에서 use_pipeline=True로 학습되면 True로 설정되고
        # save() 시 pkl에 함께 저장됨. predict.py가 load() 후 이 값을 읽어서
        # 9개 raw 입력값을 230차원으로 변환할지 판단하는 데 사용.
        self.use_pipeline = False

    def train(self, X_train, y_train, x_cols=None, scaler=None):
            """
            X_train : (n_samples, n_features)  already scaled
                    n_features = 9 (pipeline off) or 230 (pipeline on)
            y_train : (n_samples,)
            x_cols  : feature names (for importance plot)
                    ※ pipeline on이어도 x_cols는 원본 9개 컴포넌트 이름 그대로 유지됨
                        (get_static_data가 x_cols를 raw 컬럼명으로 반환하기 때문)
            scaler  : fitted scaler (saved for inference)
                    ※ pipeline on이면 230차원 기준으로 fit된 scaler가 들어옴

            ※ [DEBUG] 파이프라인별로 실제 X가 다른 데이터인지 확인하기 위한 임시 로그.
            std()가 파이프라인 종류(rdkit/chemberta/unimol)에 따라 달라지는지로
            판별함 — sum()은 StandardScaler 특성상 항상 0에 가까워 판별에 부적합.
            원인 확인 끝나면 이 5줄은 제거할 것.
            """
            # print(f"[DEBUG] X_train.shape = {X_train.shape}")
            # print(f"[DEBUG] X_train[0][:5] = {X_train[0][:5]}")
            # print(f"[DEBUG] X_train[0][100:105] = {X_train[0][100:105]}")
            # print(f"[DEBUG] X_train.sum() = {X_train.sum():.6f}")
            # print(f"[DEBUG] X_train.std() = {X_train.std():.6f}")

            self.scaler = scaler
            self.x_cols = x_cols
            print(f"[RandomForest] Training...  n_train={len(X_train)}  n_features={X_train.shape[1]}")
            self.model.fit(X_train, y_train)
            print("[RandomForest] Training complete.")

    def predict(self, X):
        """Returns (y_pred, None) — None for API consistency with GP (which returns std)."""
        return self.model.predict(X), None

    def evaluate(self, X_test, y_test):
        """Compute RMSE, R² and save evaluation plots."""
        y_pred, _ = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        print(f"[RandomForest] RMSE: {rmse:.4f}  R²: {r2:.4f}")

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        # Predicted vs Actual
        axes[0].scatter(y_test, y_pred, alpha=0.6, s=30, color="#9FE1CB")
        lims = [min(y_test.min(), y_pred.min()) - 0.2,
                max(y_test.max(), y_pred.max()) + 0.2]
        axes[0].plot(lims, lims, "r--", lw=1)
        axes[0].set_xlabel("Actual")
        axes[0].set_ylabel("Predicted")
        axes[0].set_title(f"Random Forest  (R²={r2:.3f}  RMSE={rmse:.3f})")
        axes[0].grid(True, alpha=0.3)

        # Residuals
        axes[1].hist(y_pred - y_test, bins=15, edgecolor="white", color="#9FE1CB")
        axes[1].axvline(0, color="red", lw=1, ls="--")
        axes[1].set_title("Residuals")
        axes[1].set_xlabel("Predicted - Actual")

        plt.tight_layout()
        plt.savefig(RESULT_DIR / "eval.png", dpi=120)
        plt.close()
        print(f"[RandomForest] Plot saved: {RESULT_DIR}/eval.png")

        return {
            "model" : MODEL_NAME,
            "rmse"  : round(float(rmse), 4),
            "r2"    : round(float(r2), 4),
        }

    def feature_importance(self, x_cols=None):
        cols = x_cols or self.x_cols
        if cols is None:
            print("[RandomForest] x_cols not provided, skipping feature importance.")
            return

        imp = self.model.feature_importances_

        # pipeline 적용 후 feature 수가 cols보다 많을 수 있음
        if len(cols) != len(imp):
            cols = [f"feature_{i}" for i in range(len(imp))]

        idx = np.argsort(imp)

        # 상위 20개만 표시
        if len(idx) > 20:
            idx = idx[-20:]

        fig, ax = plt.subplots(figsize=(6, 8))
        ax.barh(range(len(idx)), imp[idx], color="#9FE1CB", edgecolor="white")
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([cols[i] for i in idx], fontsize=8)
        ax.set_title("Feature Importance (Top 20)")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "feature_importance.png", dpi=120)
        plt.close()
        print(f"[RandomForest] Feature importance saved: {RESULT_DIR}/feature_importance.png")

    def get_config(self):
        """
        학습 정보 기록용 하이퍼파라미터 리포트.
        train.py의 train_model()이 result.json의 meta.hyperparams에
        이 값을 그대로 저장함.

        ※ RandomForest는 GP와 달리 하이퍼파라미터가 학습 전에 이미
          config.py에서 고정되어 있으므로, self.model에서 직접 읽어옴.
        """
        return {
            "n_estimators": self.model.n_estimators,
            "random_state": self.model.random_state,
        }

    def cross_validate(self, X, y, cv=5):
        """k-fold cross validation."""
        scores = cross_val_score(
            RandomForestRegressor(
                n_estimators=config.RF_N_ESTIMATORS,
                random_state=config.RANDOM_SEED,
                n_jobs=-1
            ),
            X, y, cv=cv, scoring="r2", n_jobs=-1
        )
        print(f"[RandomForest] CV R²: {np.round(scores, 3)}"
              f"  mean={scores.mean():.3f} ± {scores.std():.3f}")
        return scores

    def save(self, use_pipeline: bool = False):
        """
        모델을 pkl로 저장.

        use_pipeline : 학습 시 heterogeneity pipeline(SMILES·RDKit·GEM)
                       사용 여부. train.py의 train_static()에서 전달됨.
                       이 값이 pkl 안에 함께 저장되어, predict.py가
                       load() 후 동일한 전처리를 자동으로 재현할 수 있게 함.
        """
        self.use_pipeline = use_pipeline
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SAVE_PATH, "wb") as f:
            pickle.dump({
                "model"       : self.model,
                "scaler"      : self.scaler,
                "x_cols"      : self.x_cols,
                # ── 예측 시 pipeline 적용 여부 판단용 ──
                "use_pipeline": self.use_pipeline,
            }, f)
        print(f"[RandomForest] Saved: {SAVE_PATH}  (use_pipeline={self.use_pipeline})")

    def load(self):
        with open(SAVE_PATH, "rb") as f:
            data = pickle.load(f)
        self.model  = data["model"]
        self.scaler = data["scaler"]
        self.x_cols = data["x_cols"]
        # ── 구버전 pkl 호환: use_pipeline 키가 없으면 False로 간주 ──
        self.use_pipeline = data.get("use_pipeline", False)
        print(f"[RandomForest] Loaded: {SAVE_PATH}  (use_pipeline={self.use_pipeline})")


if __name__ == "__main__":
    from data_preprocess import get_static_data
    X_train, X_test, y_train, y_test, x_cols, scaler = get_static_data()
    model = RandomForestModel()
    model.train(X_train, y_train, x_cols=x_cols, scaler=scaler)
    model.evaluate(X_test, y_test)
    model.feature_importance()
    model.cross_validate(X_train, y_train)
    model.save()