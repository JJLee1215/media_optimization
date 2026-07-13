"""
smile_UniMol_gem_pipe.py
CHO 배지 Heterogeneity 처리 파이프라인 (Step 1 ~ 4) — UniMol 버전

Step 1  : SMILES 변환 (딕셔너리 조회)
Step 2  : 분자 임베딩 (UniMol 사전학습 모델, frozen)
          ※ UniMol은 SMILES를 내부적으로 3D conformer로 변환한 뒤
             그 3D 구조를 벡터화함 (unimol_tools가 이 과정을 전부 처리)
Step 2-2: GEM 벡터 (대사경로 사전 지식, 완전 고정)
Step 3  : 농도 log 스케일링 + concat
Step 4  : Mean Pooling → 배지 표현 벡터 (고정 차원)

media_pipeline.py(RDKit 버전), smile_BERTA_gem_pipe.py(ChemBERTa 버전)와
구조는 동일하고, 분자 임베딩 방식만 UniMol(512dim)로 교체.

최종 벡터 예시:
   [UniMol 512개 | 물리특성 0,0,0,0,0 | log(4.1)=1.41 | GEM 1,0,1,0,1,0,0]
   총 525차원

Usage:
  from smile_UniMol_gem_pipe import UniMolMediaPipeline
  pipeline = UniMolMediaPipeline()
  X_repr = pipeline.transform(X, feature_cols)   # (n, 9) → (n, VECTOR_DIM)

Requirements:
  pip install unimol_tools --break-system-packages
  (requirements.txt에 unimol_tools 추가 필요)

※ 첫 실행 시 UniMol 사전학습 체크포인트(수백MB)를 다운로드하므로
  시간이 걸릴 수 있음. 3D conformer 생성 실패 시(금속 이온처럼
  RDKit이 3D 좌표를 못 만드는 경우) 0벡터로 대체 처리함.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from unimol_tools import UniMolRepr


# ══════════════════════════════════════════════
# Step 1. SMILES 딕셔너리
# media_pipeline.py, smile_BERTA_gem_pipe.py와 완전히 동일
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
# Step 2-2. GEM 벡터 (대사경로 사전 지식)
# 다른 두 파이프라인과 완전히 동일
# ══════════════════════════════════════════════

GEM_PATHWAYS = [
    "Glycolysis",
    "TCA_cycle",
    "PPP",
    "AA_synthesis",
    "Energy_metabolism",
    "Cofactor_enzyme",
    "Oxidative_stress",
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
GEM_DIM            = len(GEM_PATHWAYS)


class UniMolMediaPipeline:
    """
    CHO 배지 Heterogeneity 처리 파이프라인 — UniMol 임베딩 버전

    RDKit descriptor/ChemBERTa 대신 사전학습된 UniMol(3D 구조 기반
    분자 표현 모델)을 사용. unimol_tools의 UniMolRepr이 SMILES →
    3D conformer 생성 → 임베딩까지 내부적으로 전부 처리함.

    transform(X, feature_cols) 한 번 호출로 Step 1~4 완료
    X shape: (n_samples, n_components) → (n_samples, VECTOR_DIM)
    """

    def __init__(self, eps: float = 1e-6, use_gpu: bool = False):
        self.eps = eps

        # Step 2: UniMol 로드 (사전학습, frozen)
        # remove_hs=False → 수소 원자까지 포함해서 3D 구조 생성 (기본 권장값)
        print(f"[UniMolMediaPipeline] 모델 로딩 중: unimol_tools (molecule)")
        self._clf = UniMolRepr(
            data_type="molecule",
            remove_hs=False,
            use_gpu=use_gpu,
        )

        # UniMol 기본 임베딩 차원 (cls_repr 기준, 통상 512)
        self._emb_dim   = None   # 첫 임베딩 계산 후 자동 결정
        self._embeddings = self._precompute_embeddings()
        self._emb_dim    = len(next(iter(self._embeddings.values())))
        self.vector_dim  = self._emb_dim + METAL_PHYSCHEM_DIM + 1 + GEM_DIM

        print(f"[UniMolMediaPipeline] 초기화 완료")
        print(f"  UniMol embedding : {self._emb_dim}-dim")
        print(f"  Metal physchem   : {METAL_PHYSCHEM_DIM}-dim")
        print(f"  GEM pathways     : {GEM_DIM}-dim")
        print(f"  최종 벡터 차원   : {self.vector_dim}-dim")

    # ──────────────────────────────────────────
    # Step 2: SMILES → UniMol embedding
    # unimol_tools가 3D conformer 생성부터 벡터화까지 내부 처리.
    # 3D 구조 생성에 실패하는 경우(금속 이온 등)는 예외로 잡아서
    # 0벡터로 대체 — RDKit 버전의 "mol is None → zeros" 처리와 동일한 철학.
    # ──────────────────────────────────────────
    def _smiles_to_embedding(self, smiles: str) -> np.ndarray:
        """
        unimol_tools의 get_repr()은 dict가 아니라 numpy 배열을 담은
        리스트를 반환함 — result[0]이 (emb_dim,) 형태의 임베딩 벡터.
        (버전에 따라 반환 형식이 dict일 수도 있어 두 경우 모두 방어)
        """
        try:
            result = self._clf.get_repr([smiles], return_atomic_reprs=False)
            if isinstance(result, dict) and "cls_repr" in result:
                raw = result["cls_repr"][0]
            else:
                raw = result[0]
            vec = np.array(raw, dtype=np.float32)
            return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            print(f"[UniMolMediaPipeline] 경고: '{smiles}' 임베딩 실패 ({e}) → 0벡터로 대체")
            dim = self._emb_dim if self._emb_dim else 512
            return np.zeros(dim, dtype=np.float32)

    def _precompute_embeddings(self) -> dict:
        return {
            comp: self._smiles_to_embedding(smiles)
            for comp, smiles in COMPONENT_SMILES.items()
        }

    # ──────────────────────────────────────────
    # Step 3: 단일 컴포넌트 벡터 생성
    # [UniMol | metal_physchem | log(농도) | GEM]
    # ──────────────────────────────────────────
    def _build_component_vector(self, comp: str, conc: float) -> np.ndarray:
        mol_emb   = self._embeddings[comp]                                        # (emb_dim,)
        metal_vec = METAL_PHYSCHEM.get(comp, np.zeros(METAL_PHYSCHEM_DIM, dtype=np.float32))
        log_conc  = np.array([np.log(max(conc, 0) + self.eps)], dtype=np.float32)
        gem_vec   = np.array(GEM_VECTORS[comp], dtype=np.float32)
        return np.concatenate([mol_emb, metal_vec, log_conc, gem_vec])

    # ──────────────────────────────────────────
    # Step 4: 배치 1개 → Mean Pooling
    # media_pipeline.py와 완전히 동일
    # ──────────────────────────────────────────
    def _batch_row_to_vector(self, row: np.ndarray, feature_cols: list) -> np.ndarray:
        """
        row          : (n_components,) 농도값 배열
        feature_cols : 컬럼명 리스트 (COMPONENT_SMILES 키와 매핑)

        NaN 또는 0 이하 → 해당 컴포넌트 스킵 (heterogeneity 핵심 처리)
        컴포넌트 수가 달라도 항상 동일 차원 출력
        """
        vecs = []
        for col, conc in zip(feature_cols, row):
            if col not in COMPONENT_SMILES:
                continue                        # 알 수 없는 컴포넌트 스킵
            if np.isnan(conc) or conc <= 0:
                continue                        # 없는 컴포넌트 스킵
            vecs.append(self._build_component_vector(col, float(conc)))

        if not vecs:
            return np.zeros(self.vector_dim, dtype=np.float32)

        return np.mean(vecs, axis=0).astype(np.float32)   # Mean Pooling

    # ──────────────────────────────────────────
    # 외부 인터페이스
    # ──────────────────────────────────────────
    def transform(self, X: np.ndarray, feature_cols: list) -> np.ndarray:
        """
        X            : (n_samples, n_components)  농도값 행렬
        feature_cols : 컬럼명 리스트

        Returns
        -------
        X_repr : (n_samples, VECTOR_DIM)  배지 표현 벡터
        """
        X_repr = np.array([
            self._batch_row_to_vector(row, feature_cols)
            for row in X
        ], dtype=np.float32)

        print(f"[UniMolMediaPipeline] transform 완료: {X.shape} → {X_repr.shape}")
        return X_repr


# ══════════════════════════════════════════════
# 단독 실행 테스트
# ══════════════════════════════════════════════
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import pandas as pd
    import config

    df = pd.read_csv(config.DATA_STATIC)
    drop_cols    = ["Batch_ID", "titer_final", "viab_final"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X            = df[feature_cols].values.astype(np.float32)

    pipeline = UniMolMediaPipeline()
    X_repr   = pipeline.transform(X, feature_cols)

    print(f"\n결과:")
    print(f"  입력  : {X.shape}")
    print(f"  출력  : {X_repr.shape}")
    print(f"  샘플0 GEM 차원: {X_repr[0, -GEM_DIM:]}")