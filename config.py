"""
config.py
전체 프로젝트 설정 중앙 관리

경로, 하이퍼파라미터, 모델 설정을 한 곳에서 관리
모델 추가 시 여기에만 추가하면 됨
"""

import torch
from pathlib import Path

class Config:

    # ══════════════════════════════════════════════
    # 경로 설정
    # ══════════════════════════════════════════════

    # 데이터
    DATA_DIR            = Path("data_file")
    DATA_STATIC         = DATA_DIR / "batch_table.csv"          # 정적 모델용 (GP, XGBoost 등)
    DATA_TIMESERIES     = DATA_DIR / "IndPenSim_Optimized_Final.csv"  # 시계열 모델용
    DATA_GNN_STATIC     = DATA_DIR / "gnn_m_static.csv"         # GNN 정적 배지 입력
    DATA_GNN_DYNAMIC    = DATA_DIR / "gnn_X_dynamic.csv"        # GNN 시계열 입력

    # 학습된 모델 저장
    RESULTS_MODELS_DIR  = Path("Results_Models")

    # 결과 (그래프, JSON)
    RESULTS_TT_DIR      = Path("Results_Train_Test")

    # ══════════════════════════════════════════════
    # 디바이스
    # ══════════════════════════════════════════════

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ══════════════════════════════════════════════
    # 공통 학습 설정
    # ══════════════════════════════════════════════

    TRAIN_RATIO     = 0.8       # train / val split
    TEST_SIZE       = 0.2       # train_test_split
    VAL_SIZE        = 0.1       # val split (PyTorch 모델)
    RANDOM_SEED     = 42

    # ══════════════════════════════════════════════
    # 정적 모델 (GP, XGBoost, RandomForest)
    # 입력: (n_samples, n_features) 2D
    # ══════════════════════════════════════════════

    # 공통 입력 컬럼
    STATIC_INPUT_COLS = [
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
    STATIC_TARGET_COL = "titer_final"

    # GP
    GP_N_RESTARTS   = 5

    # XGBoost
    XGB_N_ESTIMATORS = 100

    # RandomForest
    RF_N_ESTIMATORS  = 100

    # ══════════════════════════════════════════════
    # 시계열 모델 (RNN, LSTM, Transformer)
    # 입력: (n_samples, T, n_features) 3D
    # ══════════════════════════════════════════════

    TS_BATCH_COL    = "Batch ID"
    TS_FAULT_COL    = "Fault flag"
    TS_TIME_COL     = "Time (h)"
    TS_TARGET_COL   = "Penicillin concentration(P:g/L)"
    TS_SEQ_LEN      = 100       # 시점 수

    TS_INPUT_COLS   = [
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

    # RNN / LSTM 공통
    RNN_HIDDEN_SIZE = 32
    RNN_NUM_LAYERS  = 1
    RNN_EPOCHS      = 100
    RNN_LR          = 1e-3
    RNN_BATCH_SIZE  = 16

    # Transformer
    TF_D_MODEL      = 32
    TF_NHEAD        = 4
    TF_NUM_LAYERS   = 2
    TF_DROPOUT      = 0.1
    TF_EPOCHS       = 100
    TF_LR           = 1e-3
    TF_BATCH_SIZE   = 16

    # MLP
    MLP_HIDDEN_DIMS = [32, 16]
    MLP_DROPOUT     = 0.1
    MLP_EPOCHS      = 200
    MLP_LR          = 1e-3
    MLP_BATCH_SIZE  = 16

    # ══════════════════════════════════════════════
    # GNN 모델 (Model3)
    # 입력: m_static (d_static,) + X_dynamic (T, d_dynamic)
    # ══════════════════════════════════════════════

    # 데이터 차원
    GNN_N_BATCHES       = 30        # 배치 수 (더미 데이터용)
    GNN_T               = 10        # 시점 수
    GNN_D_STATIC        = 4         # m_static 차원 (glc0, glut0, Mn0, Cu0)
    GNN_D_DYN_PROCESS   = 6         # 공정변수 채널 수
    GNN_D_DYN_FEED      = 3         # feeding 채널 수
    GNN_D_DYNAMIC       = GNN_D_DYN_PROCESS + GNN_D_DYN_FEED   # = 9

    # 변수 이름 (그래프 노드 순서)
    GNN_VARIABLE_NAMES = [
        "glc", "glut", "pH", "DO", "VCD", "viab",      # 공정변수
        "feed_glc", "feed_glut", "feed_vol"             # feeding
    ]
    GNN_N = len(GNN_VARIABLE_NAMES)     # 노드 수 = 9

    # 모델 구조
    GNN_D_HIDDEN        = 64
    GNN_N_LAYERS        = 2
    GNN_MLP_HIDDEN      = 32

    # 학습
    GNN_EPOCHS          = 100
    GNN_LR              = 1e-3
    GNN_BATCH_SIZE      = 8

    # Loss 가중치
    GNN_LAMBDA_VIAB     = 0.5
    GNN_LAMBDA_GRAPH    = 1.0
    GNN_HUBER_DELTA     = 1.0

    # Domain prior A₀ (9×9)
    # 행/열 순서: glc, glut, pH, DO, VCD, viab, feed_glc, feed_glut, feed_vol
    GNN_A0 = torch.tensor([
        # glc   glut   pH    DO    VCD   viab  f_glc f_glut f_vol
        [1.0,  0.9,  0.1,  0.1,  0.9,  0.5,  0.9,  0.1,  0.0],  # glc
        [0.9,  1.0,  0.1,  0.1,  0.5,  0.5,  0.1,  0.9,  0.0],  # glut
        [0.1,  0.1,  1.0,  0.9,  0.5,  0.5,  0.0,  0.0,  0.1],  # pH
        [0.1,  0.1,  0.9,  1.0,  0.5,  0.5,  0.0,  0.0,  0.1],  # DO
        [0.9,  0.5,  0.5,  0.5,  1.0,  0.9,  0.1,  0.1,  0.0],  # VCD
        [0.5,  0.5,  0.5,  0.5,  0.9,  1.0,  0.0,  0.0,  0.0],  # viab
        [0.9,  0.1,  0.0,  0.0,  0.1,  0.0,  1.0,  0.1,  0.5],  # feed_glc
        [0.1,  0.9,  0.0,  0.0,  0.1,  0.0,  0.1,  1.0,  0.5],  # feed_glut
        [0.0,  0.0,  0.1,  0.1,  0.0,  0.0,  0.5,  0.5,  1.0],  # feed_vol
    ], dtype=torch.float32)

    # ══════════════════════════════════════════════
    # 유틸리티
    # ══════════════════════════════════════════════

    @classmethod
    def make_dirs(cls):
        """필요한 폴더 일괄 생성"""
        dirs = [
            cls.RESULTS_MODELS_DIR,
            cls.RESULTS_TT_DIR / "gp",
            cls.RESULTS_TT_DIR / "bayesian_opt",
            cls.RESULTS_TT_DIR / "xgboost",
            cls.RESULTS_TT_DIR / "random_forest",
            cls.RESULTS_TT_DIR / "mlp",
            cls.RESULTS_TT_DIR / "rnn",
            cls.RESULTS_TT_DIR / "lstm",
            cls.RESULTS_TT_DIR / "transformer",
            cls.RESULTS_TT_DIR / "gnn",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        print(f"[Config] 폴더 생성 완료")

    @classmethod
    def model_save_path(cls, model_name: str) -> Path:
        """모델 저장 경로 반환
        예: Config.model_save_path("gnn") → Results_Models/gnn_best.pt
        """
        ext = ".pt" if model_name in {"mlp", "rnn", "lstm", "transformer", "gnn"} else ".pkl"
        return cls.RESULTS_MODELS_DIR / f"{model_name}_best{ext}"

    @classmethod
    def result_dir(cls, model_name: str) -> Path:
        """결과 저장 경로 반환
        예: Config.result_dir("gnn") → Results_Train_Test/gnn/
        """
        return cls.RESULTS_TT_DIR / model_name


if __name__ == "__main__":
    cfg = Config()
    cfg.make_dirs()

    print(f"\n[경로 확인]")
    print(f"  데이터 (정적)    : {cfg.DATA_STATIC}")
    print(f"  데이터 (시계열)  : {cfg.DATA_TIMESERIES}")
    print(f"  모델 저장        : {cfg.RESULTS_MODELS_DIR}/")
    print(f"  결과 저장        : {cfg.RESULTS_TT_DIR}/")

    print(f"\n[모델별 저장 경로]")
    for m in ["gp", "xgboost", "random_forest", "mlp", "rnn", "lstm", "transformer", "gnn"]:
        print(f"  {m:<15} → {cfg.model_save_path(m)}")

    print(f"\n[GNN 설정]")
    print(f"  노드 수     : {cfg.GNN_N}")
    print(f"  변수 이름   : {cfg.GNN_VARIABLE_NAMES}")
    print(f"  d_hidden    : {cfg.GNN_D_HIDDEN}")
    print(f"  A0 shape    : {cfg.GNN_A0.shape}")