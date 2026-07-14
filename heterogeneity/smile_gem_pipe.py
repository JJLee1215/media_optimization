"""
smile_gem_pipe.py
CHO 배지 Heterogeneity 처리 파이프라인 — RDKit 버전

BaseMediaPipeline(base_pipeline.py)을 상속받아, RDKit descriptor 기반
SMILES → embedding 변환만 이 클래스가 담당함. concat/pooling/scaler/PCA는
전부 베이스 클래스가 처리.

Usage:
  from smile_gem_pipe import MediaPipeline
  pipeline = MediaPipeline(pooling_method="mean", use_pca=False, other_blocks=["gem"])
  X_train_repr = pipeline.fit_transform(X_train, feature_cols)
  X_test_repr  = pipeline.transform(X_test, feature_cols)
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_pipeline import BaseMediaPipeline, COMPONENT_SMILES


class MediaPipeline(BaseMediaPipeline):
    """RDKit descriptor 임베딩 버전."""

    def __init__(self, eps: float = 1e-6, pooling_method: str = "mean",
                 use_pca: bool = False, pca_dim: int = 30, other_blocks: list = None):
        super().__init__(eps=eps, pooling_method=pooling_method,
                          use_pca=use_pca, pca_dim=pca_dim, other_blocks=other_blocks)

        descriptor_names = [d[0] for d in Descriptors.descList]
        self._calc     = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
        self._emb_dim   = len(descriptor_names)
        self._embeddings = self._precompute_embeddings()

        print(f"[MediaPipeline] 초기화 완료")
        print(f"  RDKit descriptor : {self._emb_dim}-dim")
        print(f"  Pooling method   : {self.pooling_method}")
        print(f"  Other blocks     : {self.other_blocks}")
        print(f"  PCA              : {'ON (dim=' + str(self.pca_dim) + ')' if self.use_pca else 'OFF'}")
        print(f"  최종 벡터 차원   : {self.vector_dim}-dim")

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

    pipeline = MediaPipeline(pooling_method="mean", use_pca=False)
    X_train_repr = pipeline.fit_transform(X_train, feature_cols)
    X_test_repr  = pipeline.transform(X_test, feature_cols)

    print(f"\n결과:")
    print(f"  train : {X_train.shape} → {X_train_repr.shape}")
    print(f"  test  : {X_test.shape} → {X_test_repr.shape}")