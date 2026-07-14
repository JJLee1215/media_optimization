"""
smile_UniMol_gem_pipe.py
CHO 배지 Heterogeneity 처리 파이프라인 — UniMol 버전

BaseMediaPipeline을 상속받아, UniMol 사전학습 모델 기반
SMILES → embedding 변환만 이 클래스가 담당.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from unimol_tools import UniMolRepr

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_pipeline import BaseMediaPipeline, COMPONENT_SMILES


class UniMolMediaPipeline(BaseMediaPipeline):
    """UniMol 임베딩 버전 (3D conformer 기반)."""

    def __init__(self, eps: float = 1e-6, pooling_method: str = "mean",
                 use_pca: bool = False, pca_dim: int = 30, other_blocks: list = None,
                 use_gpu: bool = False):
        super().__init__(eps=eps, pooling_method=pooling_method,
                          use_pca=use_pca, pca_dim=pca_dim, other_blocks=other_blocks)

        print(f"[UniMolMediaPipeline] 모델 로딩 중: unimol_tools (molecule)")
        self._clf = UniMolRepr(
            data_type="molecule",
            remove_hs=False,
            use_gpu=use_gpu,
        )

        self._emb_dim   = None
        self._embeddings = self._precompute_embeddings()
        self._emb_dim    = len(next(iter(self._embeddings.values())))

        print(f"[UniMolMediaPipeline] 초기화 완료")
        print(f"  UniMol embedding : {self._emb_dim}-dim")
        print(f"  Pooling method   : {self.pooling_method}")
        print(f"  Other blocks     : {self.other_blocks}")
        print(f"  PCA              : {'ON (dim=' + str(self.pca_dim) + ')' if self.use_pca else 'OFF'}")
        print(f"  최종 벡터 차원   : {self.vector_dim}-dim")

    def _smiles_to_embedding(self, smiles: str) -> np.ndarray:
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


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import pandas as pd
    from sklearn.model_selection import train_test_split
    import config

    df = pd.read_csv(config.DATA_STATIC)
    drop_cols    = ["Batch_ID", "titer_final", "viab_final"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X            = df[feature_cols].values.astype(np.float32)

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    pipeline = UniMolMediaPipeline(pooling_method="mean", use_pca=False)
    X_train_repr = pipeline.fit_transform(X_train, feature_cols)
    X_test_repr  = pipeline.transform(X_test, feature_cols)

    print(f"\n결과:")
    print(f"  train : {X_train.shape} → {X_train_repr.shape}")
    print(f"  test  : {X_test.shape} → {X_test_repr.shape}")