"""
smile_BERTA_gem_pipe.py
CHO 배지 Heterogeneity 처리 파이프라인 — ChemBERTa 버전

BaseMediaPipeline을 상속받아, ChemBERTa 사전학습 모델 기반
SMILES → embedding 변환만 이 클래스가 담당.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModel

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_pipeline import BaseMediaPipeline, COMPONENT_SMILES

CHEMBERTA_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"


class ChemBERTaMediaPipeline(BaseMediaPipeline):
    """ChemBERTa 임베딩 버전 (frozen feature extractor)."""

    def __init__(self, eps: float = 1e-6, pooling_method: str = "mean",
                 use_pca: bool = False, pca_dim: int = 30, other_blocks: list = None,
                 device: str = None):
        super().__init__(eps=eps, pooling_method=pooling_method,
                          use_pca=use_pca, pca_dim=pca_dim, other_blocks=other_blocks)

        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"[ChemBERTaMediaPipeline] 모델 로딩 중: {CHEMBERTA_MODEL_NAME}")
        self._tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL_NAME)
        self._model     = AutoModel.from_pretrained(CHEMBERTA_MODEL_NAME).to(self.device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False

        self._emb_dim   = self._model.config.hidden_size
        self._embeddings = self._precompute_embeddings()

        print(f"[ChemBERTaMediaPipeline] 초기화 완료")
        print(f"  ChemBERTa embedding : {self._emb_dim}-dim")
        print(f"  Pooling method      : {self.pooling_method}")
        print(f"  Other blocks        : {self.other_blocks}")
        print(f"  PCA                 : {'ON (dim=' + str(self.pca_dim) + ')' if self.use_pca else 'OFF'}")
        print(f"  최종 벡터 차원      : {self.vector_dim}-dim")

    @torch.no_grad()
    def _smiles_to_embedding(self, smiles: str) -> np.ndarray:
        inputs = self._tokenizer(smiles, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask    = inputs["attention_mask"].unsqueeze(-1)

        summed = (token_embeddings * attention_mask).sum(dim=1)
        count  = attention_mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = (summed / count).squeeze(0)

        vec = mean_pooled.cpu().numpy().astype(np.float32)
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

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

    pipeline = ChemBERTaMediaPipeline(pooling_method="mean", use_pca=False)
    X_train_repr = pipeline.fit_transform(X_train, feature_cols)
    X_test_repr  = pipeline.transform(X_test, feature_cols)

    print(f"\n결과:")
    print(f"  train : {X_train.shape} → {X_train_repr.shape}")
    print(f"  test  : {X_test.shape} → {X_test_repr.shape}")