"""
media_pipeline.py
CHO 배지 Heterogeneity 처리 파이프라인 (Step 1 ~ 4)

Step 1  : SMILES 변환 (딕셔너리 조회)
Step 2  : 분자 임베딩 (RDKit descriptor, 완전 고정)
Step 2-2: GEM 벡터 (대사경로 사전 지식, 완전 고정)
Step 3  : 농도 log 스케일링 + concat
Step 4  : Mean Pooling → 배지 표현 벡터 (고정 차원)

최종 벡터 예시:
   [RDKit 217개 | 물리특성 0,0,0,0,0 | log(4.1)=1.41 | GEM 1,0,1,0,1,0,0]
   총 230차원

Usage:
  from media_pipeline import MediaPipeline
  pipeline = MediaPipeline()
  X_repr = pipeline.transform(X, feature_cols)   # (n, 9) → (n, VECTOR_DIM)
"""

# ============================================================
# CHO 배지 Heterogeneity 처리 파이프라인 핵심 개념
# ============================================================
#
# 문제:
#   배치마다 배지 컴포넌트가 다름
#   배치1: Glucose, Glutamine, Fe  (3개)
#   배치2: Glucose, Cu, Zn, Mn    (4개)
#   → 컴포넌트 수/종류가 달라 일반 ML 모델에 바로 넣을 수 없음
#
# 해결:
#   각 컴포넌트를 230차원 벡터로 변환 후 평균
#   → 컴포넌트 수/종류 달라도 항상 (배치수, 230) 출력
#
# ============================================================
# 예시 1. Glucose (농도 4.1 g/L)
# ============================================================
#
# Step 1. 이름 → SMILES (분자 구조 문자열)
#   "Glucose_0" → "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O"
#   PubChem에서 복사해온 포도당의 원자 연결 구조
#
# Step 2. SMILES → RDKit descriptor 217개 (자동 계산, 고정)
#   분자량=180.1, logP=-3.24, 극성=110.4, 수소결합수=5, ...
#   → [180.1, -3.24, 110.4, 5, ...] (217개)
#   유기분자는 값이 풍부하게 채워짐
#
# Step 2-2. 물리특성 5개 (금속 아니므로 0으로 채움)
#   Glucose는 이온이 아님 → [0, 0, 0, 0, 0]
#
# Step 3. 농도 log 변환 1개
#   log(4.1 + ε) = 1.41
#   log 쓰는 이유: 농도 범위가 0.001 ~ 200으로 넓어서
#                  그대로 쓰면 큰 값이 임베딩을 왜곡함
#
# Step 2-2. GEM 벡터 7개 (사전 지식, 고정)
#   Glucose는 해당과정(O), TCA(X), PPP(O), AA합성(X),
#             에너지대사(O), Cofactor(X), 산화스트레스(X)
#   → [1, 0, 1, 0, 1, 0, 0]
#
# 최종 Glucose 벡터:
#   [RDKit 217개 | 물리특성 0,0,0,0,0 | log(4.1)=1.41 | GEM 1,0,1,0,1,0,0]
#   총 230차원
#
# ============================================================
# 예시 2. Cu2+ (농도 0.03 g/L)
# ============================================================
#
# Step 1. 이름 → SMILES
#   "Cu_0" → "[Cu+2]"
#   구리 이온을 SMILES로 표현
#
# Step 2. SMILES → RDKit descriptor 217개 (자동 계산, 고정)
#   금속 이온은 유기분자가 아니라 대부분 0으로 채워짐
#   → [55.8, 0, 0, 0, 0, ...] (217개, 원자량 외 대부분 0)
#   RDKit이 금속 이온을 잘 표현 못하는 한계가 있음
#
# Step 2-2. 물리특성 5개 (금속이므로 직접 입력)
#   RDKit 한계를 보완하기 위해 이온 특성을 수동으로 추가
#   원자번호=29, 이온반지름=0.73Å, 전기음성도=1.90,
#   원자량=63.5, 산화수=2
#   → [29, 0.73, 1.90, 63.5, 2]
#
# Step 3. 농도 log 변환 1개
#   log(0.03 + ε) = -3.51
#
# Step 2-2. GEM 벡터 7개 (사전 지식, 고정)
#   Cu2+는 해당과정(X), TCA(X), PPP(X), AA합성(X),
#          에너지대사(O), Cofactor(O), 산화스트레스(O)
#   → [0, 0, 0, 0, 1, 1, 1]
#   Cu2+는 특정 효소의 조인자로 작동하고
#   산화스트레스(활성산소 반응)와 밀접하게 관련됨
#
# 최종 Cu2+ 벡터:
#   [RDKit 217개 | 물리특성 29,0.73,1.90,63.5,2 | log(0.03)=-3.51 | GEM 0,0,0,0,1,1,1]
#   총 230차원
#
# ============================================================
# 두 컴포넌트를 합치면 (이 배치에 Glucose, Cu만 있는 경우)
# ============================================================
#
#   Glucose → [g1, g2, ..., g230]
#   Cu2+    → [c1, c2, ..., c230]
#
#   Mean Pooling (평균):
#   → [(g1+c1)/2, (g2+c2)/2, ..., (g230+c230)/2]
#   = [230차원 벡터 1개]  ← 이 배치의 최종 배지 표현
#
#   컴포넌트가 3개든 9개든 항상 230차원으로 통일됨
#   → 모델은 항상 같은 크기의 입력을 받음
# ============================================================

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


# ══════════════════════════════════════════════
# Step 1. SMILES 딕셔너리
# 새 컴포넌트: PubChem에서 Canonical SMILES 복사 후 여기 추가
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
# CHO GEM (iCHO1766 / KEGG / BiGG) 기반
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


class MediaPipeline:
    """
    CHO 배지 Heterogeneity 처리 파이프라인

    transform(X, feature_cols) 한 번 호출로 Step 1~4 완료
    X shape: (n_samples, n_components) → (n_samples, VECTOR_DIM)
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

        # Step 2: RDKit descriptor 계산기 초기화 (고정)
        descriptor_names  = [d[0] for d in Descriptors.descList]
        self._calc        = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
        self._emb_dim     = len(descriptor_names)
        self.vector_dim   = self._emb_dim + METAL_PHYSCHEM_DIM + 1 + GEM_DIM

        # 컴포넌트 임베딩 사전 계산 (고정)
        self._embeddings  = self._precompute_embeddings()

        print(f"[MediaPipeline] 초기화 완료")
        print(f"  RDKit descriptor : {self._emb_dim}-dim")
        print(f"  Metal physchem   : {METAL_PHYSCHEM_DIM}-dim")
        print(f"  GEM pathways     : {GEM_DIM}-dim")
        print(f"  최종 벡터 차원   : {self.vector_dim}-dim")

    # ──────────────────────────────────────────
    # Step 2: SMILES → RDKit descriptor
    # ──────────────────────────────────────────
    def _smiles_to_embedding(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self._emb_dim, dtype=np.float32)
        descs = np.array(self._calc.CalcDescriptors(mol), dtype=np.float32)
        return np.nan_to_num(descs, nan=0.0, posinf=0.0, neginf=0.0)

    def _precompute_embeddings(self) -> dict:
        return {
            comp: self._smiles_to_embedding(smiles)
            for comp, smiles in COMPONENT_SMILES.items()
        }

    # ──────────────────────────────────────────
    # Step 3: 단일 컴포넌트 벡터 생성
    # [RDKit | metal_physchem | log(농도) | GEM]
    # ──────────────────────────────────────────
    def _build_component_vector(self, comp: str, conc: float) -> np.ndarray:
        mol_emb   = self._embeddings[comp]                                        # (emb_dim,)
        metal_vec = METAL_PHYSCHEM.get(comp, np.zeros(METAL_PHYSCHEM_DIM, dtype=np.float32))
        log_conc  = np.array([np.log(max(conc, 0) + self.eps)], dtype=np.float32)
        gem_vec   = np.array(GEM_VECTORS[comp], dtype=np.float32)
        return np.concatenate([mol_emb, metal_vec, log_conc, gem_vec])

    # ──────────────────────────────────────────
    # Step 4: 배치 1개 → Mean Pooling
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

        print(f"[MediaPipeline] transform 완료: {X.shape} → {X_repr.shape}")
        return X_repr


# ══════════════════════════════════════════════
# 단독 실행 테스트
# ══════════════════════════════════════════════
if __name__ == "__main__":
    import sys, os
    # 루트 디렉토리를 path에 추가
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import pandas as pd
    import config

    df = pd.read_csv(config.DATA_STATIC)
    drop_cols    = ["Batch_ID", "titer_final", "viab_final"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X            = df[feature_cols].values.astype(np.float32)

    pipeline = MediaPipeline()
    X_repr   = pipeline.transform(X, feature_cols)

    print(f"\n결과:")
    print(f"  입력  : {X.shape}")
    print(f"  출력  : {X_repr.shape}")
    print(f"  샘플0 GEM 차원: {X_repr[0, -GEM_DIM:]}")