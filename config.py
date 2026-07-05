"""
config.py
Central configuration for the CHO cell culture bioprocess ML project

Covers:
  - File paths
  - Training settings
  - Model hyperparameters
"""

import torch
from pathlib import Path


# ══════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════

DATA_DIR            = Path("data_file")
DATA_STATIC         = DATA_DIR / "batch_table_syn.csv"
DATA_TIMESERIES     = DATA_DIR / "timeseries_syn.csv"

RESULTS_MODELS_DIR  = Path("Results_Models")
RESULTS_TT_DIR      = Path("Results_Train_Test")


# ══════════════════════════════════════════════
# Device
# ══════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════
# Common training settings
# ══════════════════════════════════════════════

TRAIN_RATIO  = 0.8
TEST_SIZE    = 0.2
VAL_SIZE     = 0.1
RANDOM_SEED  = 42


# ══════════════════════════════════════════════
# Static models (GP, XGBoost, RandomForest, MLP)
# ══════════════════════════════════════════════

GP_N_RESTARTS    = 5
XGB_N_ESTIMATORS = 100
RF_N_ESTIMATORS  = 100

MLP_HIDDEN_DIMS  = [32, 16]
MLP_DROPOUT      = 0.1
MLP_EPOCHS       = 200
MLP_LR           = 1e-3
MLP_BATCH_SIZE   = 16


# ══════════════════════════════════════════════
# Time series models (RNN, LSTM, Transformer)
# ══════════════════════════════════════════════

RNN_HIDDEN_SIZE  = 32
RNN_NUM_LAYERS   = 1
RNN_EPOCHS       = 100
RNN_LR           = 1e-3
RNN_BATCH_SIZE   = 16

TF_D_MODEL       = 32
TF_NHEAD         = 4
TF_NUM_LAYERS    = 2
TF_DROPOUT       = 0.1
TF_EPOCHS        = 100
TF_LR            = 1e-3
TF_BATCH_SIZE    = 16


# ══════════════════════════════════════════════
# TCN model (Temporal Convolutional Network)
#
# TCN은 Dilated Causal Convolution을 쌓아서 시계열을 처리.
# LSTM과 달리 병렬 처리가 가능해 학습이 빠르고 안정적.
# receptive field = (kernel_size - 1) * sum(dilations)
# num_channels 리스트 길이 = 레이어 수, 각 값 = 채널(hidden) 수
# ══════════════════════════════════════════════

TCN_NUM_CHANNELS = [32, 32, 32]   # 레이어별 채널 수 (레이어 수 = len)
TCN_KERNEL_SIZE  = 3               # conv kernel 크기
TCN_DROPOUT      = 0.1
TCN_EPOCHS       = 100
TCN_LR           = 1e-3
TCN_BATCH_SIZE   = 16


# ══════════════════════════════════════════════
# GNN model (StaticTimeGNN)
# Dimensions are determined at runtime from data.py
# ══════════════════════════════════════════════

GNN_D_HIDDEN     = 64
GNN_N_LAYERS     = 2
GNN_MLP_HIDDEN   = 32
GNN_EPOCHS       = 100
GNN_LR           = 1e-3
GNN_BATCH_SIZE   = 8

GNN_LAMBDA_VIAB  = 0.5
GNN_LAMBDA_GRAPH = 1.0
GNN_HUBER_DELTA  = 1.0


# ══════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════

def make_dirs():
    """Create all output directories."""
    dirs = [
        RESULTS_MODELS_DIR,
        RESULTS_TT_DIR / "gaussian_process",
        RESULTS_TT_DIR / "xgboost",
        RESULTS_TT_DIR / "random_forest",
        RESULTS_TT_DIR / "mlp",
        RESULTS_TT_DIR / "rnn",
        RESULTS_TT_DIR / "lstm",
        RESULTS_TT_DIR / "transformer",
        RESULTS_TT_DIR / "tcn",              # ← 추가
        RESULTS_TT_DIR / "gnn",
        RESULTS_TT_DIR / "static_time_gnn",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("[Config] Directories created.")


def model_save_path(model_name: str) -> Path:
    """Return model save path. e.g. model_save_path('tcn') → Results_Models/tcn_best.pt"""
    ext = ".pt" if model_name in {
        "mlp", "rnn", "lstm", "transformer", "tcn",   # ← tcn 추가
        "gnn", "static_time_gnn"
    } else ".pkl"
    return RESULTS_MODELS_DIR / f"{model_name}_best{ext}"


def result_dir(model_name: str) -> Path:
    """Return result directory. e.g. result_dir('tcn') → Results_Train_Test/tcn/"""
    return RESULTS_TT_DIR / model_name


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

if __name__ == "__main__":
    make_dirs()

    print(f"\n[Paths]")
    print(f"  Static data     : {DATA_STATIC}")
    print(f"  Time series data: {DATA_TIMESERIES}")
    print(f"  Model outputs   : {RESULTS_MODELS_DIR}/")
    print(f"  Results         : {RESULTS_TT_DIR}/")

    print(f"\n[TCN hyperparameters]")
    print(f"  num_channels : {TCN_NUM_CHANNELS}")
    print(f"  kernel_size  : {TCN_KERNEL_SIZE}")
    print(f"  epochs       : {TCN_EPOCHS}")
    print(f"  lr           : {TCN_LR}")

    print(f"\n[GNN hyperparameters]")
    print(f"  d_hidden   : {GNN_D_HIDDEN}")
    print(f"  n_layers   : {GNN_N_LAYERS}")
    print(f"  epochs     : {GNN_EPOCHS}")
    print(f"  lr         : {GNN_LR}")

    print(f"\n[Model save paths]")
    for m in ["gaussian_process", "xgboost", "random_forest", "mlp",
              "rnn", "lstm", "transformer", "tcn", "static_time_gnn"]:
        print(f"  {m:<20} → {model_save_path(m)}")