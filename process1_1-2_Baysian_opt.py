"""
4_bayesian_opt.py
Bayesian Optimization — 다음 실험할 배지 조성 추천

흐름:
  1. 기존 데이터로 GP 학습
  2. Acquisition Function (EI) 으로 다음 후보 탐색
  3. 추천 배지 출력
  4. (시뮬레이션) 결과 추가 → 반복

CHO 데이터로 교체 시: DATA_PATH만 변경
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data_file/batch_table.csv"
OUT_DIR   = "outputs/bo"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# 입력 컬럼명 (2_make_batch_table.py SHORT 와 동일)
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


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load(path):
    df = pd.read_csv(path)
    x_cols = [c for c in X_COLS if c in df.columns]
    X = df[x_cols].values.astype(np.float64)
    y = df["titer_final"].values.astype(np.float64)
    return X, y, x_cols


# ── GP 빌드 ───────────────────────────────────────────────────────────────────
def build_gp():
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


# ── Acquisition Function: EI (Expected Improvement) ──────────────────────────
def expected_improvement(X_candidate, gp, scaler, y_best, xi=0.01):
    """
    EI = E[max(f(x) - y_best, 0)]

    y_best : 지금까지 관측된 최대 titer
    xi     : exploration 강도 (클수록 탐험적)
    """
    X_scaled = scaler.transform(X_candidate)
    mu, sigma = gp.predict(X_scaled, return_std=True)

    improvement = mu - y_best - xi
    Z = improvement / (sigma + 1e-9)
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-10] = 0.0
    return ei


# ── 다음 후보 탐색 ────────────────────────────────────────────────────────────
def suggest_next(gp, scaler, X_observed, y_observed, bounds, n_restarts=10):
    """
    EI를 최대화하는 X를 찾아 반환
    bounds: [(min, max), ...] 각 변수의 탐색 범위
    """
    y_best = y_observed.max()
    dim    = X_observed.shape[1]
    best_x, best_ei = None, -np.inf

    for _ in range(n_restarts):
        # 랜덤 시작점
        x0 = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])

        res = minimize(
            fun     = lambda x: -expected_improvement(
                          x.reshape(1, -1), gp, scaler, y_best),
            x0      = x0,
            bounds  = bounds,
            method  = "L-BFGS-B",
        )
        if -res.fun > best_ei:
            best_ei = -res.fun
            best_x  = res.x

    return best_x, best_ei


# ── 탐색 범위 자동 설정 ───────────────────────────────────────────────────────
def make_bounds(X, x_cols, margin=0.2):
    """
    학습 데이터 범위에서 ±20% 확장
    CHO 데이터에서는 실험 가능한 범위로 직접 지정 권장
    """
    bounds = []
    for i in range(X.shape[1]):
        lo = X[:, i].min()
        hi = X[:, i].max()
        span = (hi - lo) * margin
        # 변동 없는 변수(고정값)는 그 값 고정
        if span < 1e-6:
            bounds.append((lo * 0.9, lo * 1.1 + 1e-6))
        else:
            bounds.append((lo - span, hi + span))
    return bounds


# ── BO 루프 ───────────────────────────────────────────────────────────────────
def run_bo(X_init, y_init, x_cols, bounds, n_iter=5):
    """
    n_iter 번 반복하며 다음 실험 후보를 추천.
    실제 실험에서는 추천 → 실험 → 결과 입력 → 반복.
    여기서는 시뮬레이션: 실제 결과 대신 GP 예측값으로 대체.
    """
    X_obs = X_init.copy()
    y_obs = y_init.copy()
    scaler = StandardScaler()

    history = []

    print(f"\n{'iter':>5}  {'추천 배지 (상위 3 변수)':^40}  {'EI':>8}  {'현재 best titer':>15}")
    print("-" * 75)

    for i in range(1, n_iter + 1):
        scaler.fit(X_obs)
        gp = build_gp()
        gp.fit(scaler.transform(X_obs), y_obs)

        next_x, ei_val = suggest_next(gp, scaler, X_obs, y_obs, bounds)

        # 시뮬레이션: 실제 실험 대신 GP로 예측한 값 사용
        next_y_pred, next_sigma = gp.predict(
            scaler.transform(next_x.reshape(1, -1)), return_std=True
        )
        next_y = float(next_y_pred[0])

        # 상위 3개 변수만 출력 (지면 절약)
        top3 = sorted(zip(x_cols, next_x), key=lambda t: abs(t[1]), reverse=True)[:3]
        top3_str = ", ".join([f"{n}={v:.2f}" for n, v in top3])

        print(f"{i:>5}  {top3_str:<40}  {ei_val:>8.4f}  {y_obs.max():>15.3f}")

        history.append({
            "iter":    i,
            "next_x":  next_x.tolist(),
            "ei":      float(ei_val),
            "y_pred":  next_y,
            "y_sigma": float(next_sigma[0]),
            "y_best":  float(y_obs.max()),
        })

        # 새 데이터 추가
        X_obs = np.vstack([X_obs, next_x])
        y_obs = np.append(y_obs, next_y)

    return history, X_obs, y_obs, scaler, gp


# ── 결과 시각화 ───────────────────────────────────────────────────────────────
def plot_bo_history(history, out_dir):
    iters   = [h["iter"]   for h in history]
    y_bests = [h["y_best"] for h in history]
    y_preds = [h["y_pred"] for h in history]
    sigmas  = [h["y_sigma"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Best titer 추이
    axes[0].plot(iters, y_bests, "o-", color="#4B9FE0", lw=2, ms=6, label="current best")
    axes[0].fill_between(
        iters,
        [p - 2*s for p, s in zip(y_preds, sigmas)],
        [p + 2*s for p, s in zip(y_preds, sigmas)],
        alpha=0.2, color="#4B9FE0", label="±2σ of next candidate"
    )
    axes[0].set_title("BO 진행: best titer 추이")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("titer (g/L)")
    axes[0].legend(fontsize=8)

    # EI 추이
    axes[1].bar(iters, [h["ei"] for h in history], color="#9FE1CB", edgecolor="white")
    axes[1].set_title("Acquisition (EI) 추이")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("Expected Improvement")

    plt.tight_layout()
    out = f"{out_dir}/bo_history.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\n[그래프] {out}")


def print_best_recommendation(history, x_cols):
    """최종 추천 배지 출력"""
    last = history[-1]
    best_iter = max(history, key=lambda h: h["y_pred"])

    print("\n" + "="*60)
    print("  최종 추천 배지 조성")
    print("="*60)
    for col, val in zip(x_cols, best_iter["next_x"]):
        print(f"  {col:<35} {val:.4f}")
    print(f"\n  예측 titer : {best_iter['y_pred']:.3f} ± {best_iter['y_sigma']:.3f} g/L")
    print(f"  (iteration {best_iter['iter']}에서 추천)")


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    print(f"[로드] {DATA_PATH}")
    X, y, x_cols = load(DATA_PATH)
    print(f"  X: {X.shape}  Y: {y.shape}")
    print(f"  초기 best titer: {y.max():.3f}")

    bounds = make_bounds(X, x_cols)
    print(f"\n[탐색 범위]")
    for col, (lo, hi) in zip(x_cols, bounds):
        print(f"  {col:<35} [{lo:.3f}, {hi:.3f}]")

    print("\n[Bayesian Optimization 시작]  (5 iterations)")
    history, X_final, y_final, scaler, gp = run_bo(
        X, y, x_cols, bounds, n_iter=5
    )

    plot_bo_history(history, OUT_DIR)
    print_best_recommendation(history, x_cols)

    print(f"\n[완료] outputs/bo/ 에서 결과 확인")
    print("CHO 데이터 교체 후: bounds를 실험 가능 범위로 직접 수정 권장")