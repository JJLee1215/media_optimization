import torch
from torch.utils.data import Dataset, DataLoader, random_split
from process3_1_config import Config

class BioprocessDataset(Dataset):
    """
    배치 배양 데이터셋

    각 샘플:
        m_static   (d_static,)          정적 배지 조성
        X_dynamic  (T, d_dynamic)       시계열 공정변수 + feeding
        y_titer    scalar               최종 titer (g/L)
        y_viab     scalar               최종 viability (0~1)
    """

    def __init__(self, m_static, X_dynamic, y_titer, y_viab):
        self.m_static  = m_static    # (n_batches, d_static)
        self.X_dynamic = X_dynamic   # (n_batches, T, d_dynamic)
        self.y_titer   = y_titer     # (n_batches,)
        self.y_viab    = y_viab      # (n_batches,)

    def __len__(self):
        return len(self.m_static)

    def __getitem__(self, idx):
        return (
            self.m_static[idx],
            self.X_dynamic[idx],
            self.y_titer[idx],
            self.y_viab[idx],
        )


def generate_dummy_data(cfg: Config):
    """
    더미 데이터 생성
    실제 데이터로 교체할 때 이 함수만 수정하면 됨

    m_static 채널:   [glc0, glut0, Mn0, Cu0]
    X_dynamic 채널:  [glc, glut, pH, DO, VCD, viab, feed_glc, feed_glut, feed_vol]
    """
    torch.manual_seed(42)
    n = cfg.n_batches

    # ── m_static: 배지 초기 조성 ──────────────────────
    # glc0: 3~6 g/L, glut0: 1~3 g/L, Mn: 0.01~0.1, Cu: 0.01~0.05
    m_static = torch.stack([
        torch.FloatTensor(n).uniform_(3.0, 6.0),    # glc0
        torch.FloatTensor(n).uniform_(1.0, 3.0),    # glut0
        torch.FloatTensor(n).uniform_(0.01, 0.1),   # Mn0
        torch.FloatTensor(n).uniform_(0.01, 0.05),  # Cu0
    ], dim=1)   # (n, 4)

    # ── X_dynamic: 시계열 공정변수 ────────────────────
    X_dynamic = torch.zeros(n, cfg.T, cfg.d_dynamic)

    for i in range(n):
        glc0 = m_static[i, 0].item()

        for t in range(cfg.T):
            day = t + 1
            feed_day = (t == 4) or (t == 8)   # Day5, Day9에 feeding

            # 공정변수 (간단한 시뮬레이션)
            X_dynamic[i, t, 0] = max(0.1, glc0 - 0.35 * day + (1.5 if feed_day else 0) + torch.randn(1).item() * 0.1)   # glc
            X_dynamic[i, t, 1] = max(0.1, m_static[i,1] - 0.15 * day + torch.randn(1).item() * 0.05)                    # glut
            X_dynamic[i, t, 2] = 7.0 + torch.randn(1).item() * 0.1                                                       # pH
            X_dynamic[i, t, 3] = max(10, 45 - 1.5 * day + torch.randn(1).item() * 2)                                     # DO
            X_dynamic[i, t, 4] = 0.5 + 0.5 * day + torch.randn(1).item() * 0.1                                           # VCD
            X_dynamic[i, t, 5] = max(0.5, 0.95 - 0.02 * day + torch.randn(1).item() * 0.01)                              # viability

            # feeding 채널
            X_dynamic[i, t, 6] = 2.0 if feed_day else 0.0    # feed_glc (g)
            X_dynamic[i, t, 7] = 0.5 if feed_day else 0.0    # feed_glut (g)
            X_dynamic[i, t, 8] = 50.0 if feed_day else 0.0   # feed_vol (mL)

    # ── 타깃: titer, viability ────────────────────────
    # 배지 조성과 공정 마지막 상태에서 결정 (간단한 선형 관계)
    y_titer = (
        0.5 * m_static[:, 0] +          # glc0 기여
        0.3 * m_static[:, 1] +          # glut0 기여
        0.2 * X_dynamic[:, -1, 4] +     # 최종 VCD 기여
        torch.randn(n) * 0.2
    ).clamp(min=0.1)   # (n,)

    y_viab = (
        0.85 + 0.1 * m_static[:, 0] / 6.0 +
        torch.randn(n) * 0.03
    ).clamp(0.5, 1.0)   # (n,)

    return m_static, X_dynamic, y_titer, y_viab


def get_dataloaders(cfg: Config):
    """train / val DataLoader 반환"""
    m_static, X_dynamic, y_titer, y_viab = generate_dummy_data(cfg)

    dataset = BioprocessDataset(m_static, X_dynamic, y_titer, y_viab)

    n_train = int(len(dataset) * cfg.train_ratio)
    n_val   = len(dataset) - n_train

    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    cfg = Config()
    train_loader, val_loader = get_dataloaders(cfg)

    m, X, yt, yv = next(iter(train_loader))
    print(f"m_static  shape: {m.shape}")     # (batch, 4)
    print(f"X_dynamic shape: {X.shape}")     # (batch, 10, 9)
    print(f"y_titer   shape: {yt.shape}")    # (batch,)
    print(f"y_viab    shape: {yv.shape}")    # (batch,)