"""
3_gp_model.py
Gaussian Process Regression으로 titer 예측

데이터 적을 때 (n < 100) 적합
- 예측값 + 불확실성(신뢰구간) 동시 출력
- CHO 데이터로 교체 시 DATA_PATH만 변경
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import json

DATA_PATH = "data_file/batch_table.csv"
OUT_DIR   = "outputs/gp"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# 2_make_batch_table.py 의 SHORT 딕셔너리와 동일
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
SHORT = {c: c.split("(")[0].strip() for c in INPUT_COLS}
X_COLS = list(SHORT.values())   # batch_table.csv 의 X 컬럼명


def load(path):
    df = pd.read_csv(path)
    # batch_table.csv 컬럼: batch_id, <X_COLS...>, titer_final
    x_cols = [c for c in X_COLS if c in df.columns]
    X = df[x_cols].values.astype(np.float32)
    y = df["titer_final"].values.astype(np.float32)
    print(f"  X: {X.shape}   Y: {y.shape}")
    print(f"  X 컬럼: {x_cols}")
    return X, y, x_cols


def build_gp():
    """
    커널 구성:
      ConstantKernel  → 전체 스케일
      RBF             → 부드러운 상관관계
      WhiteKernel     → 노이즈
    """
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        random_state=42,
    )


def evaluate(model, X_test, y_test, scaler, out_dir):
    y_pred, y_std = model.predict(scaler.transform(X_test), return_std=True)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # 예측 vs 실제 + 불확실성
    axes[0].errorbar(y_test, y_pred, yerr=2*y_std,
                     fmt="o", alpha=0.6, capsize=3, ms=5, label="±2σ")
    lims = [min(y_test.min(), y_pred.min()) - 2,
            max(y_test.max(), y_pred.max()) + 2]
    axes[0].plot(lims, lims, "r--", lw=1)
    axes[0].set_xlabel("Actual titer")
    axes[0].set_ylabel("Predicted titer")
    axes[0].set_title(f"GP 예측  (R²={r2:.3f})")
    axes[0].legend(fontsize=8)

    # 불확실성 분포
    axes[1].hist(y_std, bins=15, edgecolor="white", color="#4B9FE0")
    axes[1].set_title("예측 불확실성 (σ) 분포")
    axes[1].set_xlabel("σ")
    plt.tight_layout()
    out = f"{out_dir}/gp_eval.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  [그래프] {out}")

    return {"rmse": round(float(rmse), 4), "r2": round(float(r2), 4)}


def plot_feature_importance(model, x_cols, out_dir):
    try:
        ls = model.kernel_.k1.k2.length_scale
        if np.isscalar(ls):
            print("  (단일 length_scale — 변수별 중요도 불가)")
            return
        importance = 1.0 / ls
        importance /= importance.sum()

        fig, ax = plt.subplots(figsize=(6, 5))
        idx = np.argsort(importance)
        ax.barh(range(len(idx)), importance[idx], color="#4B9FE0")
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([x_cols[i] for i in idx], fontsize=8)
        ax.set_title("변수 중요도 (1/length_scale)")
        ax.set_xlabel("상대적 중요도")
        plt.tight_layout()
        out = f"{out_dir}/feature_importance.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  [그래프] {out}")
    except Exception as e:
        print(f"  (중요도 그래프 생략: {e})")


if __name__ == "__main__":
    print(f"[로드] {DATA_PATH}")
    X, y, x_cols = load(DATA_PATH)

    # train / test 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  train={len(X_train)}  test={len(X_test)}")

    # 스케일링 (Y는 GP 내부 normalize_y=True 로 처리)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # GP 학습
    print("\n[GP 학습 중...]")
    gp = build_gp()
    gp.fit(X_train_s, y_train)
    print(f"  학습된 커널: {gp.kernel_}")

    # 테스트 평가
    print("\n[테스트 평가]")
    result = evaluate(gp, X_test, y_test, scaler, OUT_DIR)

    # 5-Fold CV
    print("\n[5-Fold 교차검증]")
    X_s = scaler.fit_transform(X)
    cv_scores = cross_val_score(build_gp(), X_s, y, cv=5,
                                scoring="r2", n_jobs=-1)
    print(f"  R² per fold : {np.round(cv_scores, 3)}")
    print(f"  R² mean     : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # 변수 중요도
    print("\n[변수 중요도]")
    plot_feature_importance(gp, x_cols, OUT_DIR)

    # 결과 저장
    result["cv_r2_mean"] = round(float(cv_scores.mean()), 4)
    result["cv_r2_std"]  = round(float(cv_scores.std()), 4)
    with open(f"{OUT_DIR}/result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[완료] outputs/gp/ 에서 결과 확인")
    print("CHO 데이터 교체 시: data_file/batch_table.csv 를 같은 포맷으로 교체 후 재실행")