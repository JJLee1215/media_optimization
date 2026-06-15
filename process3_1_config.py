import torch

class Config:
    # ── 디바이스 ──────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 데이터 ───────────────────────────────
    n_batches       = 30        # 배치 수
    T               = 10        # 시점 수 (Day 수)
    d_static        = 4         # m_static 차원 (glc0, glut0, Mn0, Cu0)
    d_dyn_process   = 6         # 공정변수 채널 수 (glc, glut, pH, DO, VCD, viab)
    d_dyn_feed      = 3         # feeding 채널 수 (feed_glc, feed_glut, feed_vol)
    d_dynamic       = d_dyn_process + d_dyn_feed   # = 9

    # ── 변수 이름 (그래프 노드 순서) ──────────
    variable_names  = [
        "glc", "glut", "pH", "DO", "VCD", "viab",   # 공정변수
        "feed_glc", "feed_glut", "feed_vol"           # feeding
    ]
    N = len(variable_names)     # 노드 수 = 9

    # ── 모델 ─────────────────────────────────
    d_hidden        = 64        # hidden 차원
    n_gnn_layers    = 2         # GNN 메시지 패싱 횟수
    mlp_hidden      = 32        # MLP hidden 차원

    # ── Loss 가중치 ───────────────────────────
    lambda_viab     = 0.5       # L_viab 가중치
    lambda_graph    = 1.0       # L_graph 가중치
    huber_delta     = 1.0       # Huber loss delta

    # ── 학습 ─────────────────────────────────
    epochs          = 100
    lr              = 1e-3
    batch_size      = 8         # 30개 배치 → 8개씩 미니배치
    train_ratio     = 0.8       # train/val split

    # ── A₀ domain prior (9×9) ────────────────
    # 행/열 순서: glc, glut, pH, DO, VCD, viab, feed_glc, feed_glut, feed_vol
    A0 = torch.tensor([
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