"""
smile_BERTA_gem_pipe.py
CHO 배지 Heterogeneity 처리 파이프라인 (Step 1 ~ 4) — ChemBERTa 버전

Step 1  : SMILES 변환 (딕셔너리 조회)
Step 2  : 분자 임베딩 (ChemBERTa 사전학습 모델, frozen)
Step 2-2: GEM 벡터 (대사경로 사전 지식, 완전 고정)
Step 3  : 농도 log 스케일링 + concat
Step 4  : Mean Pooling → 배지 표현 벡터 (고정 차원)

media_pipeline.py(RDKit 버전)와 구조는 동일하고, 분자 임베딩 방식만
RDKit descriptor(217dim) 대신 ChemBERTa 사전학습 모델(768dim)을 사용.

최종 벡터 예시:
   [ChemBERTa 768개 | 물리특성 0,0,0,0,0 | log(4.1)=1.41 | GEM 1,0,1,0,1,0,0]
   총 781차원

Usage:
  from smile_BERTA_gem_pipe import ChemBERTaMediaPipeline
  pipeline = ChemBERTaMediaPipeline()
  X_repr = pipeline.transform(X, feature_cols)   # (n, 9) → (n, VECTOR_DIM)

Requirements:
  pip install transformers --break-system-packages
  (requirements.txt에 transformers 추가 필요)
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModel


# ══════════════════════════════════════════════
# Step 1. SMILES 딕셔너리
# media_pipeline.py와 완전히 동일 (같은 컴포넌트를 쓰므로 중복 유지)
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
# media_pipeline.py와 완전히 동일
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

# ══════════════════════════════════════════════
# ChemBERTa 모델 설정
# ══════════════════════════════════════════════

CHEMBERTA_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"


class ChemBERTaMediaPipeline:
    """
    CHO 배지 Heterogeneity 처리 파이프라인 — ChemBERTa 임베딩 버전

    RDKit descriptor 대신 사전학습된 ChemBERTa(SMILES 기반 언어모델)를
    사용해 분자 임베딩을 생성. 모델 가중치는 학습 중 고정(frozen)됨 —
    파인튜닝하지 않고 특징 추출기로만 사용.

    transform(X, feature_cols) 한 번 호출로 Step 1~4 완료
    X shape: (n_samples, n_components) → (n_samples, VECTOR_DIM)
    """

    def __init__(self, eps: float = 1e-6, device: str = None):
        self.eps    = eps
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Step 2: ChemBERTa 로드 (사전학습, frozen)
        print(f"[ChemBERTaMediaPipeline] 모델 로딩 중: {CHEMBERTA_MODEL_NAME}")
        self._tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL_NAME)
        self._model     = AutoModel.from_pretrained(CHEMBERTA_MODEL_NAME).to(self.device)
        self._model.eval()                        # 학습 모드 아님 (frozen feature extractor)
        for p in self._model.parameters():
            p.requires_grad = False                # 그래디언트 계산 비활성화 (메모리/속도)

        self._emb_dim   = self._model.config.hidden_size   # 보통 768
        self.vector_dim = self._emb_dim + METAL_PHYSCHEM_DIM + 1 + GEM_DIM

        # 컴포넌트 임베딩 사전 계산 (고정 — RDKit 버전과 동일한 패턴)
        self._embeddings = self._precompute_embeddings()

        print(f"[ChemBERTaMediaPipeline] 초기화 완료")
        print(f"  ChemBERTa embedding : {self._emb_dim}-dim")
        print(f"  Metal physchem      : {METAL_PHYSCHEM_DIM}-dim")
        print(f"  GEM pathways        : {GEM_DIM}-dim")
        print(f"  최종 벡터 차원      : {self.vector_dim}-dim")

    # ──────────────────────────────────────────
    # Step 2: SMILES → ChemBERTa embedding
    # mean pooling: 토큰별 hidden state를 attention mask 기준으로 평균
    # (ChemBERTa는 RoBERTa 계열이라 BERT의 [CLS] 풀링 대신 mean pooling이
    #  문장/분자 표현으로 더 안정적이라고 알려져 있음)
    # ──────────────────────────────────────────
    @torch.no_grad()
    def _smiles_to_embedding(self, smiles: str) -> np.ndarray:
        inputs = self._tokenizer(smiles, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        token_embeddings = outputs.last_hidden_state              # (1, seq_len, emb_dim)
        attention_mask    = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)

        summed = (token_embeddings * attention_mask).sum(dim=1)
        count  = attention_mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = (summed / count).squeeze(0)                  # (emb_dim,)

        vec = mean_pooled.cpu().numpy().astype(np.float32)
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    def _precompute_embeddings(self) -> dict:
        return {
            comp: self._smiles_to_embedding(smiles)
            for comp, smiles in COMPONENT_SMILES.items()
        }

    # ──────────────────────────────────────────
    # Step 3: 단일 컴포넌트 벡터 생성
    # [ChemBERTa | metal_physchem | log(농도) | GEM]
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

        print(f"[ChemBERTaMediaPipeline] transform 완료: {X.shape} → {X_repr.shape}")
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

    pipeline = ChemBERTaMediaPipeline()
    X_repr   = pipeline.transform(X, feature_cols)

    print(f"\n결과:")
    print(f"  입력  : {X.shape}")
    print(f"  출력  : {X_repr.shape}")
    print(f"  샘플0 GEM 차원: {X_repr[0, -GEM_DIM:]}")