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
        self.model  = self._build()
        self.scaler = None
        self.x_cols = None

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
        X_train : (n_train, 9)  already scaled
        y_train : (n_train,)
        x_cols  : feature names (for importance plot)
        scaler  : fitted StandardScaler (saved for inference)
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

    def save(self):
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SAVE_PATH, "wb") as f:
            pickle.dump({
                "model"  : self.model,
                "scaler" : self.scaler,
                "x_cols" : self.x_cols,
            }, f)
        print(f"[GaussianProcess] Saved: {SAVE_PATH}")

    def load(self):
        with open(SAVE_PATH, "rb") as f:
            data = pickle.load(f)
        self.model  = data["model"]
        self.scaler = data["scaler"]
        self.x_cols = data["x_cols"]
        print(f"[GaussianProcess] Loaded: {SAVE_PATH}")


if __name__ == "__main__":
    from data_preprocess import get_static_data
    X_train, X_test, y_train, y_test, x_cols, scaler = get_static_data()

    model = GaussianProcessModel()
    model.train(X_train, y_train, x_cols=x_cols, scaler=scaler)
    model.evaluate(X_test, y_test)
    model.cross_validate(X_train, y_train)
    model.save()