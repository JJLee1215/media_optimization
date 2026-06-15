"""
Models/Model_GP.py
Gaussian Process Regression

train()    → GP 학습
predict()  → 예측값 + 불확실도(σ) 반환
evaluate() → RMSE, R² 계산 + 시각화
save()     → pkl 저장
load()     → pkl 로드
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from config import Config

cfg = Config()
MODEL_NAME  = "gp"
SAVE_PATH   = cfg.model_save_path(MODEL_NAME)
RESULT_DIR  = cfg.result_dir(MODEL_NAME)


class GPModel:
    def __init__(self):
        self.model  = self._build()
        self.scaler = None    # data.py에서 받아옴

    def _build(self):
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=cfg.GP_N_RESTARTS,
            normalize_y=True,
            random_state=cfg.RANDOM_SEED,
        )

    def train(self, X_train, y_train, scaler=None):
        """
        X_train: (n, d)  이미 스케일링된 입력
        y_train: (n,)
        """
        self.scaler = scaler
        print(f"[GP] 학습 중...  train={len(X_train)}")
        self.model.fit(X_train, y_train)
        print(f"[GP] 학습 완료  커널: {self.model.kernel_}")

    def predict(self, X):
        """
        Returns:
            y_pred: (n,)
            y_std:  (n,)  불확실도
        """
        y_pred, y_std = self.model.predict(X, return_std=True)
        return y_pred, y_std

    def evaluate(self, X_test, y_test):
        """RMSE, R² 계산 + 시각화 저장"""
        y_pred, y_std = self.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        print(f"[GP] RMSE: {rmse:.4f}  R²: {r2:.4f}")

        RESULT_DIR.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        # 예측 vs 실제 + 불확실도
        axes[0].errorbar(y_test, y_pred, yerr=2*y_std,
                         fmt="o", alpha=0.6, capsize=3, ms=5, label="±2σ")
        lims = [min(y_test.min(), y_pred.min()) - 1,
                max(y_test.max(), y_pred.max()) + 1]
        axes[0].plot(lims, lims, "r--", lw=1)
        axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
        axes[0].set_title(f"GP  (R²={r2:.3f}  RMSE={rmse:.3f})")
        axes[0].legend(fontsize=8)

        # 불확실도 분포
        axes[1].hist(y_std, bins=15, edgecolor="white", color="#4B9FE0")
        axes[1].set_title("Uncertainty (σ) distribution")
        axes[1].set_xlabel("σ")

        plt.tight_layout()
        plt.savefig(RESULT_DIR / "eval.png", dpi=120)
        plt.close()
        print(f"[GP] 결과 저장: {RESULT_DIR}/eval.png")

        return {"model": MODEL_NAME, "rmse": round(float(rmse), 4), "r2": round(float(r2), 4)}

    def cross_validate(self, X, y, cv=5):
        """5-Fold 교차검증"""
        scores = cross_val_score(self._build(), X, y, cv=cv,
                                 scoring="r2", n_jobs=-1)
        print(f"[GP] CV R²: {np.round(scores, 3)}  mean={scores.mean():.3f} ± {scores.std():.3f}")
        return scores

    def save(self):
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SAVE_PATH, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)
        print(f"[GP] 저장: {SAVE_PATH}")

    def load(self):
        with open(SAVE_PATH, "rb") as f:
            data = pickle.load(f)
        self.model  = data["model"]
        self.scaler = data["scaler"]
        print(f"[GP] 로드: {SAVE_PATH}")


if __name__ == "__main__":
    from data import get_static_data
    X_train, X_test, y_train, y_test, x_cols, scaler = get_static_data(use_syn=True)

    gp = GPModel()
    gp.train(X_train, y_train, scaler=scaler)
    result = gp.evaluate(X_test, y_test)
    gp.save()
    print(result)