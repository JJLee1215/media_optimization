import torch
import torch.nn as nn
import torch.nn.functional as F
from process3_1_config import Config


# ══════════════════════════════════════════════════════
# 1. Static Encoder
#    m_static (d_static,) → h0, c0 (d_hidden,)
# ══════════════════════════════════════════════════════
class StaticEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.h0_mlp = nn.Sequential(
            nn.Linear(cfg.d_static, cfg.d_hidden),
            nn.Tanh()
        )
        self.c0_mlp = nn.Sequential(
            nn.Linear(cfg.d_static, cfg.d_hidden),
            nn.Tanh()
        )

    def forward(self, m_static):
        """
        입력: m_static  (batch, d_static)
        출력: h0        (1, batch, d_hidden)
              c0        (1, batch, d_hidden)
        """
        h0 = self.h0_mlp(m_static).unsqueeze(0)   # (1, batch, d_hidden)
        c0 = self.c0_mlp(m_static).unsqueeze(0)   # (1, batch, d_hidden)
        return h0, c0


# ══════════════════════════════════════════════════════
# 2. Dynamic Encoder
#    X_dynamic (T, d_dynamic) → H_dynamic (T, d_hidden)
#    h0, c0 로 LSTM 초기화
# ══════════════════════════════════════════════════════
class DynamicEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.d_dynamic,
            hidden_size=cfg.d_hidden,
            num_layers=1,
            batch_first=True
        )

    def forward(self, X_dynamic, h0, c0):
        """
        입력: X_dynamic  (batch, T, d_dynamic)
              h0         (1, batch, d_hidden)
              c0         (1, batch, d_hidden)
        출력: H_dynamic  (batch, T, d_hidden)
        """
        H_dynamic, _ = self.lstm(X_dynamic, (h0, c0))
        return H_dynamic


# ══════════════════════════════════════════════════════
# 3. Cross-Attention
#    Q = H_dynamic, K/V = m_static
#    H_dynamic* (T, d_hidden) — 배지 맥락 반영
# ══════════════════════════════════════════════════════
class CrossAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.Wq = nn.Linear(cfg.d_hidden, cfg.d_hidden, bias=False)
        self.Wk = nn.Linear(cfg.d_static,  cfg.d_hidden, bias=False)
        self.Wv = nn.Linear(cfg.d_static,  cfg.d_hidden, bias=False)
        self.norm = nn.LayerNorm(cfg.d_hidden)
        self.scale = cfg.d_hidden ** 0.5

    def forward(self, H_dynamic, m_static):
        """
        입력: H_dynamic  (batch, T, d_hidden)
              m_static   (batch, d_static)
        출력: H_dynamic* (batch, T, d_hidden)
        """
        Q = self.Wq(H_dynamic)                     # (batch, T, d_hidden)
        K = self.Wk(m_static).unsqueeze(1)         # (batch, 1, d_hidden)
        V = self.Wv(m_static).unsqueeze(1)         # (batch, 1, d_hidden)

        # attention score: (batch, T, 1)
        attn = torch.softmax(
            torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1
        )

        # context: (batch, T, d_hidden)
        context = torch.bmm(attn, V)

        # residual + LayerNorm
        out = self.norm(H_dynamic + context)
        return out


# ══════════════════════════════════════════════════════
# 4. Graph Construction
#    V = H_dynamic*[-1]  (N, d_hidden)
#    Ã = sigmoid(vᵢWₐvⱼᵀ) ⊙ M_causal + A₀
# ══════════════════════════════════════════════════════
class GraphConstruction(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.Wa = nn.Linear(cfg.d_hidden, cfg.d_hidden, bias=False)

        # A₀: domain prior (고정, 학습 안 함)
        self.register_buffer("A0", cfg.A0)

        # M_causal: 생물학적으로 불가능한 엣지 마스킹
        # 현재는 모두 허용 (1.0) → 필요시 특정 위치 0으로 설정
        M = torch.ones(cfg.N, cfg.N)
        self.register_buffer("M_causal", M)

    def forward(self, H_dynamic_star):
        """
        입력: H_dynamic_star  (batch, T, d_hidden)
        출력: V  (batch, N, d_hidden)   노드 특성
              A  (batch, N, N)           adjacency
        """
        # 마지막 시점 추출 → 노드 특성
        V = H_dynamic_star[:, -1, :]           # (batch, d_hidden)
        # d_dynamic 채널 = N개 변수이므로 reshape
        # 실제로는 각 변수의 hidden을 분리해야 함
        # 여기서는 d_hidden을 N등분하거나, 별도 projection 사용
        # 단순화: V를 N개 노드로 확장 (batch, N, d_hidden)
        batch = V.shape[0]
        V = V.unsqueeze(1).expand(-1, self.A0.shape[0], -1)   # (batch, N, d_hidden)

        # adjacency 학습
        V_proj = self.Wa(V)                    # (batch, N, d_hidden)
        # (batch, N, N)
        A_raw = torch.bmm(V_proj, V_proj.transpose(1, 2))
        A = torch.sigmoid(A_raw)

        # causal constraint + domain prior
        A_tilde = A * self.M_causal + self.A0.unsqueeze(0)
        A_tilde = A_tilde.clamp(0, 1)

        return V, A_tilde


# ══════════════════════════════════════════════════════
# 5. GNN Reasoning
#    H⁽ˡ⁺¹⁾ = σ( D̃⁻¹ Ã H⁽ˡ⁾ W⁽ˡ⁾ )
# ══════════════════════════════════════════════════════
class GNNLayer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.W = nn.Linear(cfg.d_hidden, cfg.d_hidden)

    def forward(self, H, A_tilde):
        """
        입력: H        (batch, N, d_hidden)
              A_tilde  (batch, N, N)
        출력: H_next   (batch, N, d_hidden)
        """
        # degree normalization: D̃⁻¹
        deg = A_tilde.sum(dim=-1, keepdim=True).clamp(min=1e-6)   # (batch, N, 1)
        A_norm = A_tilde / deg                                      # (batch, N, N)

        # message passing
        agg = torch.bmm(A_norm, H)    # (batch, N, d_hidden)
        out = F.relu(self.W(agg))
        return out


class GNN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.layers = nn.ModuleList([
            GNNLayer(cfg) for _ in range(cfg.n_gnn_layers)
        ])

    def forward(self, V, A_tilde):
        """
        입력: V        (batch, N, d_hidden)
              A_tilde  (batch, N, N)
        출력: H_final  (batch, N, d_hidden)
        """
        H = V
        for layer in self.layers:
            H = layer(H, A_tilde)
        return H


# ══════════════════════════════════════════════════════
# 6. Output Heads
#    mean pooling → MLP_titer, MLP_viab
# ══════════════════════════════════════════════════════
class OutputHead(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.mlp_titer = nn.Sequential(
            nn.Linear(cfg.d_hidden, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Linear(cfg.mlp_hidden, 1)
        )
        self.mlp_viab = nn.Sequential(
            nn.Linear(cfg.d_hidden, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Linear(cfg.mlp_hidden, 1),
            nn.Sigmoid()   # viability: 0~1
        )

    def forward(self, H_final):
        """
        입력: H_final  (batch, N, d_hidden)
        출력: mu_titer (batch,)
              y_viab   (batch,)
        """
        # mean pooling: (batch, d_hidden)
        h_pool = H_final.mean(dim=1)

        mu_titer = self.mlp_titer(h_pool).squeeze(-1)   # (batch,)
        y_viab   = self.mlp_viab(h_pool).squeeze(-1)    # (batch,)

        return mu_titer, y_viab


# ══════════════════════════════════════════════════════
# 7. 전체 Model3
# ══════════════════════════════════════════════════════
class Model3(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.static_encoder    = StaticEncoder(cfg)
        self.dynamic_encoder   = DynamicEncoder(cfg)
        self.cross_attention   = CrossAttention(cfg)
        self.graph_construction = GraphConstruction(cfg)
        self.gnn               = GNN(cfg)
        self.output_head       = OutputHead(cfg)

    def forward(self, m_static, X_dynamic):
        """
        입력: m_static   (batch, d_static)
              X_dynamic  (batch, T, d_dynamic)
        출력: mu_titer   (batch,)
              y_viab     (batch,)
              A_tilde    (batch, N, N)   → loss 계산용
        """
        # ① Static encoder
        h0, c0 = self.static_encoder(m_static)

        # ② Dynamic encoder
        H_dynamic = self.dynamic_encoder(X_dynamic, h0, c0)

        # ③ Cross-attention
        H_dynamic_star = self.cross_attention(H_dynamic, m_static)

        # ④ Graph construction
        V, A_tilde = self.graph_construction(H_dynamic_star)

        # ⑤ GNN
        H_final = self.gnn(V, A_tilde)

        # ⑥ Output
        mu_titer, y_viab = self.output_head(H_final)

        return mu_titer, y_viab, A_tilde


if __name__ == "__main__":
    cfg   = Config()
    model = Model3(cfg)

    # 더미 입력으로 forward pass 확인
    batch = 4
    m  = torch.randn(batch, cfg.d_static)
    X  = torch.randn(batch, cfg.T, cfg.d_dynamic)

    mu, viab, A = model(m, X)
    print(f"mu_titer : {mu.shape}")    # (4,)
    print(f"y_viab   : {viab.shape}")  # (4,)
    print(f"A_tilde  : {A.shape}")     # (4, 9, 9)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 수: {total_params:,}")