"""
heterogeneity/base_pipeline.py
RDKit / ChemBERTa / UniMol 세 파이프라인의 공통 로직을 모은 베이스 클래스.

각 서브클래스(MediaPipeline, ChemBERTaMediaPipeline, UniMolMediaPipeline)는
__init__에서 self._embeddings(성분별 임베딩 딕셔너리)와 self._emb_dim만
채우면, concat/pooling/scaler/PCA는 전부 여기서 공통 처리됨.

Step 1  : SMILES 딕셔너리 (COMPONENT_SMILES) — 공통 상수
Step 2  : 분자 임베딩 — 서브클래스가 self._embeddings로 채워둠
Step 2-2: GEM 벡터, Metal physchem — 공통 상수
Step 3  : 농도 log 스케일링 + concat (컴포넌트 단위)
          ※ other_blocks로 어떤 부가 블록(log_conc/metal_physchem/gem)을
            concat에 포함시킬지 선택 가능. None이면 셋 다 포함(기본값).
Step 4  : Pooling — mean 또는 mean+weighted+max+count
Step 5  : StandardScaler + PCA — train만 fit, test는 transform만

other_blocks : list[str] | None
  포함 가능한 값: "log_conc", "metal_physchem", "gem"
  None이면 기본값으로 셋 다 포함 (기존 동작과의 하위 호환)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ══════════════════════════════════════════════
# Step 1. SMILES 딕셔너리 (세 파이프라인 공통)
# ══════════════════════════════════════════════

COMPONENT_SMILES = {
    "Glucose_0"    : "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O",
    "Glutamine_0"  : "C(CC(=O)N)[C@@H](C(=O)O)N",
    "Asparagine_0" : "C([C@@H](C(=O)O)N)C(=O)N",
    "Lactate_0"    : "CC(O)C(=O)O",
    "Ammonia_0"    : "N",
    "Cu_0"         : "[Cu+2]",
    "Zn_0"         : "[Zn+2]",
    "Mn_0"         : "[Mn+2]",
    "Fe_0"         : "[Fe+2]",
}

# ══════════════════════════════════════════════
# Step 2-2. GEM 벡터 (대사경로 사전 지식, 세 파이프라인 공통)
# ══════════════════════════════════════════════

GEM_PATHWAYS = [
    "Glycolysis", "TCA_cycle", "PPP", "AA_synthesis",
    "Energy_metabolism", "Cofactor_enzyme", "Oxidative_stress",
]

GEM_VECTORS = {
    "Glucose_0"    : [1, 0, 1, 0, 1, 0, 0],
    "Glutamine_0"  : [0, 1, 0, 1, 1, 0, 0],
    "Asparagine_0" : [0, 0, 0, 1, 0, 0, 0],
    "Lactate_0"    : [1, 1, 0, 0, 1, 0, 0],
    "Ammonia_0"    : [0, 0, 0, 1, 0, 0, 1],
    "Cu_0"         : [0, 0, 0, 0, 1, 1, 1],
    "Zn_0"         : [0, 0, 0, 0, 0, 1, 1],
    "Mn_0"         : [0, 1, 0, 0, 0, 1, 0],
    "Fe_0"         : [0, 1, 0, 0, 1, 1, 1],
}

# Trace metal 물리화학 보완 벡터
# (원자번호, 이온반지름, 전기음성도, 원자량, 산화수)
METAL_PHYSCHEM = {
    "Cu_0" : np.array([29, 0.73, 1.90, 63.5, 2], dtype=np.float32),
    "Zn_0" : np.array([30, 0.74, 1.65, 65.4, 2], dtype=np.float32),
    "Mn_0" : np.array([25, 0.83, 1.55, 54.9, 2], dtype=np.float32),
    "Fe_0" : np.array([26, 0.78, 1.83, 55.8, 2], dtype=np.float32),
}
METAL_PHYSCHEM_DIM = 5
GEM_DIM             = len(GEM_PATHWAYS)

ALL_BLOCKS = ["log_conc", "metal_physchem", "gem"]


class BaseMediaPipeline:
    """
    RDKit/ChemBERTa/UniMol 공통 파이프라인 로직.

    서브클래스는 __init__에서 아래 두 속성만 채우면 됨:
      self._embeddings : {comp_name: np.ndarray} — 성분별 사전학습 임베딩
      self._emb_dim     : 임베딩 1개의 차원 (int)

    나머지(concat, pooling, scaler, PCA)는 이 클래스가 전부 처리.
    """

    def __init__(self, eps: float = 1e-6, pooling_method: str = "mean",
                 use_pca: bool = False, pca_dim: int = 30, other_blocks: list = None):
        self.eps            = eps
        self.pooling_method = pooling_method   # "mean" | "multi_stat"
        self.use_pca        = use_pca
        self.pca_dim         = pca_dim
        # None이면 기존 동작(셋 다 포함)과 동일하게 처리
        self.other_blocks    = other_blocks if other_blocks is not None else list(ALL_BLOCKS)

        self._embeddings = {}   # 서브클래스가 채움
        self._emb_dim     = 0    # 서브클래스가 채움

        self.scaler = StandardScaler()
        self.pca    = PCA(n_components=pca_dim) if use_pca else None
        self._is_fitted = False

    # ──────────────────────────────────────────
    # 부가 블록 하나당 차원 (other_blocks에 포함된 것만 합산)
    # ──────────────────────────────────────────
    @property
    def _extra_dim(self) -> int:
        extra = 0
        if "metal_physchem" in self.other_blocks:
            extra += METAL_PHYSCHEM_DIM
        if "log_conc" in self.other_blocks:
            extra += 1
        if "gem" in self.other_blocks:
            extra += GEM_DIM
        return extra

    @property
    def pooled_dim(self) -> int:
        """Pooling 직후(PCA 적용 전) 차원."""
        if self.pooling_method == "multi_stat":
            # mean + weighted_mean + max 는 emb_dim만 3배, extra는 그대로 1번 + count 1
            return self._emb_dim * 3 + self._extra_dim + 1
        return self._emb_dim + self._extra_dim

    @property
    def vector_dim(self) -> int:
        """최종 출력 차원 (PCA 적용 시 pca_dim, 아니면 pooled_dim)."""
        if self.use_pca:
            return self.pca_dim
        return self.pooled_dim

    # ──────────────────────────────────────────
    # Step 3: 단일 컴포넌트 벡터 생성
    # [embedding | (선택적) metal_physchem | (선택적) log(농도) | (선택적) GEM]
    # ──────────────────────────────────────────
    def _build_component_vector(self, comp: str, conc: float) -> np.ndarray:
        parts = [self._embeddings[comp]]

        if "metal_physchem" in self.other_blocks:
            metal_vec = METAL_PHYSCHEM.get(comp, np.zeros(METAL_PHYSCHEM_DIM, dtype=np.float32))
            parts.append(metal_vec)

        if "log_conc" in self.other_blocks:
            log_conc = np.array([np.log(max(conc, 0) + self.eps)], dtype=np.float32)
            parts.append(log_conc)

        if "gem" in self.other_blocks:
            gem_vec = np.array(GEM_VECTORS[comp], dtype=np.float32)
            parts.append(gem_vec)

        return np.concatenate(parts)

    # ──────────────────────────────────────────
    # Step 4: Pooling — mean 또는 mean+weighted+max+count
    # ──────────────────────────────────────────
    def _pool(self, vecs: list, concs: list) -> np.ndarray:
        vecs = np.array(vecs)

        if self.pooling_method == "multi_stat":
            weights = np.array(concs, dtype=np.float32)
            weights = weights / weights.sum()
            mean_pool     = vecs.mean(axis=0)
            weighted_pool = (vecs * weights[:, None]).sum(axis=0)
            max_pool      = vecs.max(axis=0)
            n_components  = np.array([len(vecs)], dtype=np.float32)
            return np.concatenate([mean_pool, weighted_pool, max_pool, n_components])

        return vecs.mean(axis=0)   # 기본값: mean

    def _batch_row_to_vector(self, row: np.ndarray, feature_cols: list) -> np.ndarray:
        """
        row          : (n_components,) 농도값 배열
        feature_cols : 컬럼명 리스트 (COMPONENT_SMILES 키와 매핑)

        NaN 또는 0 이하 → 해당 컴포넌트 스킵 (heterogeneity 핵심 처리)
        """
        vecs, concs = [], []
        for col, conc in zip(feature_cols, row):
            if col not in COMPONENT_SMILES:
                continue
            if np.isnan(conc) or conc <= 0:
                continue
            vecs.append(self._build_component_vector(col, float(conc)))
            concs.append(float(conc))

        if not vecs:
            return np.zeros(self.pooled_dim, dtype=np.float32)

        return self._pool(vecs, concs).astype(np.float32)

    def _pool_batch(self, X: np.ndarray, feature_cols: list) -> np.ndarray:
        """X(농도 행렬) 전체를 pooling까지만 적용 (scaler/PCA 전 단계)."""
        return np.array([
            self._batch_row_to_vector(row, feature_cols)
            for row in X
        ], dtype=np.float32)

    # ──────────────────────────────────────────
    # Step 5: StandardScaler + PCA
    # fit은 train에서만, transform은 train/test 둘 다.
    # ──────────────────────────────────────────
    def fit_transform(self, X: np.ndarray, feature_cols: list) -> np.ndarray:
        pooled = self._pool_batch(X, feature_cols)
        scaled = self.scaler.fit_transform(pooled)
        result = self.pca.fit_transform(scaled) if self.use_pca else scaled
        self._is_fitted = True
        print(f"[{self.__class__.__name__}] fit_transform 완료: {X.shape} → {result.shape}"
              f"  (pooling={self.pooling_method}, pca={self.use_pca}, other_blocks={self.other_blocks})")
        return result

    def transform(self, X: np.ndarray, feature_cols: list) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                f"[{self.__class__.__name__}] transform() 호출 전에 fit_transform()이 "
                f"먼저 호출되어야 합니다 (train 데이터로 scaler/PCA를 학습해야 함)."
            )
        pooled = self._pool_batch(X, feature_cols)
        scaled = self.scaler.transform(pooled)
        result = self.pca.transform(scaled) if self.use_pca else scaled
        print(f"[{self.__class__.__name__}] transform 완료: {X.shape} → {result.shape}")
        return result