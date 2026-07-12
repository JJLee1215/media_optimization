"""
Models/gaussian_process.py
Gaussian Process Regression

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
     이 클래스 자체는 입력 차원에 무관하게 동작 (sklearn 모델 특성)

Target:
  y_train : (n_train,)   titer_final
  y_test  : (n_test,)

Source: batch_table_syn.csv  via data_preprocess.get_static_data()

Output:
  y_pred  : (n,)   predicted titer
  y_std   : (n,)   uncertainty (σ) — unique to GP

Methods:
  train()          fit GP model
  predict()        return predictions + uncertainty
  evaluate()       compute RMSE, R² + save plots
  cross_validate() k-fold cross validation
  save() / load()  persist model to disk

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
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

MODEL_NAME = "gaussian_process"
SAVE_PATH  = config.model_save_path(MODEL_NAME)
RESULT_DIR = config.result_dir(MODEL_NAME)


class GaussianProcessModel:
    def __init__(self):
        self.model        = self._build()
        self.scaler       = None
        self.x_cols       = None
        # ── heterogeneity pipeline 사용 여부 (기본 False) ──
        # train_static()에서 use_pipeline=True로 학습되면 True로 설정되고
        # save() 시 pkl에 함께 저장됨. predict.py가 load() 후 이 값을 읽어서
        # 9개 raw 입력값을 230차원으로 변환할지 판단하는 데 사용.
        self.use_pipeline = False

    def _build(self):
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=config.GP_N_RESTARTS,
            normalize_y=True,
            random_state=config.RANDOM_SEED,
        )

    def train(self, X_train, y_train, x_cols=None, scaler=None):
        """
        X_train : (n_train, 9) or (n_train, 230)  already scaled
                  9   = raw concentration features (pipeline off)
                  230 = media representation vector (pipeline on)
        y_train : (n_train,)
        x_cols  : feature names (for importance plot)
                  ※ pipeline on이어도 x_cols는 원본 9개 컴포넌트 이름 그대로 유지됨
                    (get_static_data가 x_cols를 raw 컬럼명으로 반환하기 때문)
        scaler  : fitted StandardScaler (saved for inference)
                  ※ pipeline on이면 230차원 기준으로 fit된 scaler가 들어옴
        """
        self.scaler = scaler
        self.x_cols = x_cols
        print(f"[GaussianProcess] Training...  n_train={len(X_train)}  n_features={X_train.shape[1]}")
        self.model.fit(X_train, y_train)
        print(f"[GaussianProcess] Training complete.")
        print(f"[GaussianProcess] Learned kernel: {self.model.kernel_}")

    def predict(self, X):
        """
        Returns:
          y_pred : (n,)   predicted titer
          y_std  : (n,)   uncertainty (σ) — unique to GP
        """
        y_pred, y_std = self.model.predict(X, return_std=True)
        return y_pred, y_std

    def evaluate(self, X_test, y_test):
        """Compute RMSE, R² and save evaluation plots."""
        y_pred, y_std = self.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        print(f"[GaussianProcess] RMSE: {rmse:.4f}  R²: {r2:.4f}")

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        # Predicted vs Actual + uncertainty
        axes[0].errorbar(y_test, y_pred, yerr=2 * y_std,
                         fmt="o", alpha=0.6, capsize=3, ms=5,
                         color="#1D9E75", label="±2σ")
        lims = [min(y_test.min(), y_pred.min()) - 0.2,
                max(y_test.max(), y_pred.max()) + 0.2]
        axes[0].plot(lims, lims, "r--", lw=1)
        axes[0].set_xlabel("Actual")
        axes[0].set_ylabel("Predicted")
        axes[0].set_title(f"Gaussian Process  (R²={r2:.3f}  RMSE={rmse:.3f})")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Uncertainty distribution
        axes[1].hist(y_std, bins=15, edgecolor="white", color="#534AB7")
        axes[1].set_title("Uncertainty (σ) distribution")
        axes[1].set_xlabel("σ")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(RESULT_DIR / "eval.png", dpi=120)
        plt.close()
        print(f"[GaussianProcess] Plot saved: {RESULT_DIR}/eval.png")

        return {
            "model": MODEL_NAME,
            "rmse" : round(float(rmse), 4),
            "r2"   : round(float(r2),   4),
        }

    def cross_validate(self, X, y, cv=5):
            """k-fold cross validation."""
            scores = cross_val_score(
                self._build(), X, y,
                cv=cv, scoring="r2", n_jobs=-1
            )
            print(f"[GaussianProcess] CV R²: {np.round(scores, 3)}"
                f"  mean={scores.mean():.3f} ± {scores.std():.3f}")
            return scores

    def get_config(self):
        """
        학습 정보 기록용 하이퍼파라미터 리포트.
        train.py의 train_model()이 result.json의 meta.hyperparams에
        이 값을 그대로 저장함.

        ※ kernel_은 model.fit() 이후에만 존재하는 학습된(fitted) 커널이라,
          학습 전에 호출하면 KeyError 대신 fallback 값을 반환하도록 방어함.
        """
        return {
            "kernel"               : str(self.model.kernel_) if hasattr(self.model, "kernel_") else None,
            "n_restarts_optimizer" : config.GP_N_RESTARTS,
            "normalize_y"          : True,
            "random_state"         : config.RANDOM_SEED,
        }


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
        print(f"[GaussianProcess] Saved: {SAVE_PATH}  (use_pipeline={self.use_pipeline})")

    def load(self):
        with open(SAVE_PATH, "rb") as f:
            data = pickle.load(f)
        self.model  = data["model"]
        self.scaler = data["scaler"]
        self.x_cols = data["x_cols"]
        # ── 구버전 pkl 호환: use_pipeline 키가 없으면 False로 간주 ──
        self.use_pipeline = data.get("use_pipeline", False)
        print(f"[GaussianProcess] Loaded: {SAVE_PATH}  (use_pipeline={self.use_pipeline})")


if __name__ == "__main__":
    from data_preprocess import get_static_data
    X_train, X_test, y_train, y_test, x_cols, scaler = get_static_data()

    model = GaussianProcessModel()
    model.train(X_train, y_train, x_cols=x_cols, scaler=scaler)
    model.evaluate(X_test, y_test)
    model.cross_validate(X_train, y_train)
    model.save()