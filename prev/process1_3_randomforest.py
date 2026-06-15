"""
6_random_forest_model.py
Random Forest로 titer 예측
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json, time

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "data_file/batch_table.csv"
OUT_DIR   = "outputs/random_forest"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

INPUT_COLS = [
    "Aeration rate(Fg:L/h)",
    "Agitator RPM(RPM:RPM)",
    "Sugar feed rate(Fs:L/h)",
    "Acid flow rate(Fa:L/h)",
    "Base flow rate(Fb:L/h)",
    "Heating/cooling water flow rate(Fc:L/h)",
    "Heating water flow rate(Fh:L/h)",
    "Water for injection/dilution(Fw:L/h)",
    "PAA flow(Fpaa:PAA flow (L/h))",
    "Oil flow(Foil:L/hr)",
]
X_COLS = [c.split("(")[0].strip() for c in INPUT_COLS]


def load(path):
    df = pd.read_csv(path)
    x_cols = [c for c in X_COLS if c in df.columns]
    X = df[x_cols].values.astype(np.float64)
    y = df["titer_final"].values.astype(np.float64)
    print(f"  X: {X.shape}   Y: {y.shape}")
    return X, y, x_cols


def evaluate(model, X_test, y_test, out_dir):
    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    print(f"\n[테스트 결과]")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].scatter(y_test, y_pred, alpha=0.6, s=30)
    lims = [min(y_test.min(), y_pred.min()) - 2,
            max(y_test.max(), y_pred.max()) + 2]
    axes[0].plot(lims, lims, "r--", lw=1)
    axes[0].set_xlabel("Actual titer")
    axes[0].set_ylabel("Predicted titer")
    axes[0].set_title(f"Random Forest  (R²={r2:.3f})")

    axes[1].hist(y_pred - y_test, bins=15, edgecolor="white", color="#9FE1CB")
    axes[1].axvline(0, color="red", lw=1, linestyle="--")
    axes[1].set_title("Residuals")
    axes[1].set_xlabel("Predicted - Actual")

    plt.tight_layout()
    out = f"{out_dir}/eval.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[그래프] {out}")
    return rmse, r2


def plot_feature_importance(model, x_cols, out_dir):
    imp = model.feature_importances_
    idx = np.argsort(imp)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(range(len(idx)), imp[idx], color="#9FE1CB")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([x_cols[i] for i in idx], fontsize=8)
    ax.set_title("Feature importance")
    ax.set_xlabel("importance")
    plt.tight_layout()
    out = f"{out_dir}/feature_importance.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[그래프] {out}")


if __name__ == "__main__":
    print(f"[로드] {DATA_PATH}")
    X, y, x_cols = load(DATA_PATH)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y, test_size=0.2, random_state=42
    )
    print(f"  train={len(X_train)}  test={len(X_test)}")

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"\n[학습 완료]  {time.time()-t0:.2f}초")

    rmse, r2 = evaluate(model, X_test, y_test, OUT_DIR)
    plot_feature_importance(model, x_cols, OUT_DIR)

    # 5-Fold CV
    print("\n[5-Fold 교차검증]")
    cv = cross_val_score(model, X_s, y, cv=5, scoring="r2", n_jobs=-1)
    print(f"  R² per fold : {np.round(cv, 3)}")
    print(f"  R² mean     : {cv.mean():.3f} ± {cv.std():.3f}")

    result = {"rmse": round(float(rmse), 4), "r2": round(float(r2), 4),
              "cv_r2_mean": round(float(cv.mean()), 4),
              "cv_r2_std":  round(float(cv.std()), 4)}
    with open(f"{OUT_DIR}/result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[완료] outputs/random_forest/ 에서 결과 확인")