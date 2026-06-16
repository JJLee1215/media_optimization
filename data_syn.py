"""
data_syn.py
Synthetic (dummy) data generation for CHO cell culture bioprocess

Standalone file — independent from config.py
Edit the SETTINGS section below to customize data structure.

Generated data:
  1. Static model data  (batch_table_syn.csv)
     - Initial media composition (m_static)
     - (n_samples, n_features) 2D
     - For GP, XGBoost, RandomForest, MLP + GNN m_static

  2. Time series model data  (timeseries_syn.csv)
     - (n_batches x seq_len, n_features) long format
     - For RNN, LSTM, Transformer + GNN X_dynamic
       X_dynamic = media concentrations + feeding events + process variables
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ══════════════════════════════════════════════
# SETTINGS — edit here to customize data structure
# ══════════════════════════════════════════════

DATA_DIR = Path("data_file")
SEED     = 42

# ── Static data settings ──────────────────────
STATIC_N_SAMPLES = 100
STATIC_SAVE_NAME = "batch_table_syn.csv"

# Initial media composition (m_static)
# {column_name: (min, max, unit)}
STATIC_FEATURES = {
    # Carbon / Nitrogen sources
    "Glucose_0"      : (3.0,   6.0,   "g/L"),
    "Glutamine_0"    : (1.0,   3.0,   "g/L"),
    "Asparagine_0"   : (50.0,  150.0, "mg/L"),
    # Metabolites (initial)
    "Lactate_0"      : (0.0,   0.5,   "g/L"),
    "Ammonia_0"      : (0.0,   0.2,   "mmol/L"),
    # Trace metals
    "Cu_0"           : (0.01,  0.05,  "mg/L"),
    "Zn_0"           : (0.01,  0.10,  "mg/L"),
    "Mn_0"           : (0.005, 0.05,  "mg/L"),
    "Fe_0"           : (0.05,  0.20,  "mg/L"),
}

# ── Time series data settings ─────────────────
TS_N_BATCHES  = 100
TS_SEQ_LEN    = 14
TS_SAVE_NAME  = "timeseries_syn.csv"

TS_BATCH_COL  = "Batch ID"
TS_TIME_COL   = "Time (day)"
TS_FAULT_COL  = "Fault flag"
TS_TARGET_COL = "Titer (g/L)"

# d_dyn_media: media component concentrations over time
# Same components as m_static — measured as consumption trajectory
# {column_name: (initial_min, initial_max, consumption_rate_per_day)}
MEDIA_FEATURES = {
    "Glucose_conc"    : (3.0,   6.0,   0.08),   # g/L
    "Glutamine_conc"  : (1.0,   3.0,   0.06),   # g/L
    "Asparagine_conc" : (50.0,  150.0, 0.05),   # mg/L
    "Lactate_conc"    : (0.0,   0.5,   -0.15),  # g/L  — accumulates (negative consumption)
    "Ammonia_conc"    : (0.0,   0.2,   -0.05),  # mmol/L — accumulates
    "Cu_conc"         : (0.01,  0.05,  0.01),   # mg/L
    "Zn_conc"         : (0.01,  0.10,  0.01),   # mg/L
    "Mn_conc"         : (0.005, 0.05,  0.005),  # mg/L
    "Fe_conc"         : (0.05,  0.20,  0.01),   # mg/L
}

# d_dyn_feed: feeding events
# Amount added at each feeding day (0 on non-feeding days)
FEED_DAYS = [3, 6, 9]   # 0-indexed (Day 4, 7, 10)
FEED_FEATURES = {
    "feed_Glucose"    : 2.0,    # g
    "feed_Glutamine"  : 0.5,    # g
    "feed_Asparagine" : 20.0,   # mg
    "feed_vol"        : 50.0,   # mL
}

# d_dyn_process: process control variables
# {column_name: (min, max)}
PROCESS_FEATURES = {
    "pH"                              : (6.8,  7.2),
    "DO"                              : (20.0, 60.0),
    "Temperature"                     : (36.0, 37.5),
    "VCD"                             : (0.5,  20.0),
    "Viability"                       : (0.7,  1.0),
    "Aeration rate"                   : (0.5,  2.0),
    "Agitator RPM"                    : (50.0, 200.0),
    "Sugar feed rate"                 : (0.01, 0.5),
    "Acid flow rate"                  : (0.0,  0.1),
    "Base flow rate"                  : (0.0,  0.1),
    "Heating/cooling water flow rate" : (0.0,  5.0),
    "Heating water flow rate"         : (0.0,  2.0),
    "Water for injection/dilution"    : (0.0,  0.5),
    "PAA flow"                        : (0.0,  0.05),
    "Oil flow"                        : (0.0,  0.02),
}


# ══════════════════════════════════════════════
# 1. Static model data + GNN m_static
# ══════════════════════════════════════════════

def make_static_data() -> pd.DataFrame:
    """
    Generates batch-level static data (initial media composition).

    Columns:
      batch_id
      [STATIC_FEATURES]   initial media composition  <- X (m_static)
      titer_final                                     <- y
      viab_final                                      <- y (GNN)
    """
    np.random.seed(SEED)
    n = STATIC_N_SAMPLES

    data = {"batch_id": range(1, n + 1)}
    col_vals = {}
    for col, (lo, hi, *_) in STATIC_FEATURES.items():
        col_vals[col] = np.random.uniform(lo, hi, n)
        data[col] = col_vals[col]

    df = pd.DataFrame(data)

    # Target: titer_final
    df["titer_final"] = (
        0.4 * col_vals["Glucose_0"]
        + 0.3 * col_vals["Glutamine_0"]
        + 0.1 * col_vals["Asparagine_0"] / 100
        - 0.2 * col_vals["Lactate_0"]
        - 0.1 * col_vals["Ammonia_0"]
        + np.random.normal(0, 0.3, n)
    ).clip(0.1)

    # Target: viab_final
    df["viab_final"] = (
        0.90
        - 0.05 * col_vals["Lactate_0"]
        - 0.03 * col_vals["Ammonia_0"]
        + np.random.normal(0, 0.02, n)
    ).clip(0.5, 1.0)

    save_path = DATA_DIR / STATIC_SAVE_NAME
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"[Static data] Saved: {save_path}  shape: {df.shape}")
    print(f"  m_static features ({len(STATIC_FEATURES)}): {list(STATIC_FEATURES.keys())}")
    print(f"  targets: titer_final, viab_final")
    return df


# ══════════════════════════════════════════════
# 2. Time series model data + GNN X_dynamic
# ══════════════════════════════════════════════

def make_timeseries_data() -> pd.DataFrame:
    """
    Generates time series data in long format.
    Each row = one (batch, time step).

    X_dynamic columns:
      [MEDIA_FEATURES]    media concentrations over time  (d_dyn_media)
      [FEED_FEATURES]     feeding events, 0 on non-feed days (d_dyn_feed)
      [PROCESS_FEATURES]  process control variables       (d_dyn_process)

    Target:
      Titer (g/L) — recorded at last time step only
    """
    np.random.seed(SEED)
    rows = []

    for batch_id in range(1, TS_N_BATCHES + 1):

        # Initial media concentrations for this batch
        current_media = {
            col: np.random.uniform(lo, hi)
            for col, (lo, hi, *_) in MEDIA_FEATURES.items()
        }

        # Initial process variable baselines
        init_process = {
            col: np.random.uniform(lo, hi)
            for col, (lo, hi) in PROCESS_FEATURES.items()
        }

        titer_final = float(
            0.4 * current_media["Glucose_conc"]
            + 0.3 * current_media["Glutamine_conc"]
            + np.random.uniform(0.5, 2.0)
        )

        for t in range(TS_SEQ_LEN):
            is_feed_day = t in FEED_DAYS
            row = {
                TS_BATCH_COL : batch_id,
                TS_TIME_COL  : t + 1,
                TS_FAULT_COL : 0,
            }

            # ── d_dyn_media ──────────────────────
            for col, (lo, hi, rate) in MEDIA_FEATURES.items():
                if is_feed_day and rate > 0:
                    current_media[col] *= 1.3    # feeding boosts concentration
                else:
                    current_media[col] -= abs(rate) * current_media[col]  # consumption

                current_media[col] = max(current_media[col], 0.001)
                row[col] = round(
                    current_media[col] + np.random.normal(0, current_media[col] * 0.02), 4
                )

            # ── d_dyn_feed ───────────────────────
            for col, amount in FEED_FEATURES.items():
                row[col] = amount if is_feed_day else 0.0

            # ── d_dyn_process ────────────────────
            for col, (lo, hi) in PROCESS_FEATURES.items():
                row[col] = float(np.clip(
                    init_process[col] + np.random.normal(0, (hi - lo) * 0.02),
                    lo, hi
                ))

            # Target
            row[TS_TARGET_COL] = round(titer_final, 4) if t == TS_SEQ_LEN - 1 else 0.0
            rows.append(row)

    df = pd.DataFrame(rows)

    save_path = DATA_DIR / TS_SAVE_NAME
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    d_media   = len(MEDIA_FEATURES)
    d_feed    = len(FEED_FEATURES)
    d_process = len(PROCESS_FEATURES)

    print(f"[Time series data] Saved: {save_path}  shape: {df.shape}")
    print(f"  d_dyn_media   ({d_media}): {list(MEDIA_FEATURES.keys())}")
    print(f"  d_dyn_feed    ({d_feed}):  {list(FEED_FEATURES.keys())}")
    print(f"  d_dyn_process ({d_process}): {list(PROCESS_FEATURES.keys())}")
    print(f"  d_dynamic total: {d_media + d_feed + d_process}")
    print(f"  Feeding days: {[d+1 for d in FEED_DAYS]}")
    return df


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Synthetic Data Generation — CHO Cell Culture Bioprocess")
    print("=" * 60)

    print("\n[1] Static data (GP, XGBoost, RF, MLP + GNN m_static)")
    make_static_data()

    print("\n[2] Time series data (RNN, LSTM, Transformer + GNN X_dynamic)")
    make_timeseries_data()

    print("\n" + "=" * 60)
    print("Done. Check data_file/ folder.")
    print("To use real data, modify data.py only.")
    print("=" * 60)