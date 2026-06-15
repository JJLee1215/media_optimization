"""
Models/Model_XGBoost.py
XGBoost Regression
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from config import Config

cfg = Config()
MODEL_NAME = "xgboost"
SAVE_PATH  = cfg.model_save_path(MODEL_NAME)
RESULT_DIR = cfg.result_dir(MODEL_NAME)


class XGBoostModel:
    def __init__(self):
        self.model  = XGBRegressor(
            n_estimators=cfg.XGB_N_ESTIMATORS,
            random_state=cfg.RANDOM_SEED,
            verbosity=0, n_jobs=-1
        )
        self.scaler = None

    def train(self, X_train, y_train, scaler=None):
        self.scaler = scaler
        print(f"[XGBoost] 학습 중...  train={len(X_train)}")
        self.model.fit(X_train, y_train)
        print("[XGBoost] 학습 완료")

    def predict(self, X):
        return self.model.predict(X), None   # (y_pred, None)

    def evaluate(self, X_test, y_test):
        y_pred, _ = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        print(f"[XGBoost] RMSE: {rmse:.4f}  R²: {r2:.4f}")

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].scatter(y_test, y_pred, alpha=0.6, s=30)
        lims = [min(y_test.min(), y_pred.min())-1, max(y_test.max(), y_pred.max())+1]
        axes[0].plot(lims, lims, "r--", lw=1)
        axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
        axes[0].set_title(f"XGBoost  (R²={r2:.3f}  RMSE={rmse:.3f})")

        axes[1].hist(y_pred - y_test, bins=15, edgecolor="white", color="#4B9FE0")
        axes[1].axvline(0, color="red", lw=1, ls="--")
        axes[1].set_title("Residuals")

        plt.tight_layout()
        plt.savefig(RESULT_DIR / "eval.png", dpi=120)
        plt.close()
        print(f"[XGBoost] 결과 저장: {RESULT_DIR}/eval.png")
        return {"model": MODEL_NAME, "rmse": round(float(rmse), 4), "r2": round(float(r2), 4)}

    def feature_importance(self, x_cols):
        imp = self.model.feature_importances_
        idx = np.argsort(imp)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(range(len(idx)), imp[idx], color="#4B9FE0")
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([x_cols[i] for i in idx], fontsize=8)
        ax.set_title("Feature Importance")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "feature_importance.png", dpi=120)
        plt.close()

    def cross_validate(self, X, y, cv=5):
        scores = cross_val_score(self.model, X, y, cv=cv, scoring="r2", n_jobs=-1)
        print(f"[XGBoost] CV R²: {np.round(scores, 3)}  mean={scores.mean():.3f} ± {scores.std():.3f}")
        return scores

    def save(self):
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SAVE_PATH, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)
        print(f"[XGBoost] 저장: {SAVE_PATH}")

    def load(self):
        with open(SAVE_PATH, "rb") as f:
            data = pickle.load(f)
        self.model  = data["model"]
        self.scaler = data["scaler"]
        print(f"[XGBoost] 로드: {SAVE_PATH}")


if __name__ == "__main__":
    from data import get_static_data
    X_train, X_test, y_train, y_test, x_cols, scaler = get_static_data(use_syn=True)
    model = XGBoostModel()
    model.train(X_train, y_train, scaler=scaler)
    model.evaluate(X_test, y_test)
    model.feature_importance(x_cols)
    model.save()