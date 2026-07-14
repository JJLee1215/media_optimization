"""
heterogeneity/_registry.py
파이프라인 레지스트리 — RDKit / ChemBERTa / UniMol 인스턴스 생성·캐싱·차원 조회를
한 곳에서 관리.

※ 캐시 키가 (embedding_model, pooling_method, use_pca, pca_dim, other_blocks) 조합.
  other_blocks까지 캐시 키에 포함해야, "concat 블록을 뭘 켰는지"에 따라
  다른 차원의 인스턴스가 서로 캐시를 잘못 공유하지 않음.
"""

from heterogeneity.base_pipeline import ALL_BLOCKS

_pipeline_cache = {}


def _normalize_blocks(other_blocks):
    """캐시 키로 쓰기 위해 리스트를 정렬된 튜플로 정규화. None이면 전체 기본값."""
    blocks = other_blocks if other_blocks is not None else list(ALL_BLOCKS)
    return tuple(sorted(blocks))


def get_pipeline(embedding_model: str = "rdkit", pooling_method: str = "mean",
                  use_pca: bool = False, pca_dim: int = 30, other_blocks: list = None):
    """
    embedding_model : "rdkit" | "chemberta" | "unimol"
    pooling_method  : "mean" | "multi_stat"
    use_pca         : PCA 적용 여부
    pca_dim         : use_pca=True일 때 목표 차원
    other_blocks    : ["log_conc", "metal_physchem", "gem"] 중 포함할 것들.
                       None이면 셋 다 포함(기본값).

    Returns: 파이프라인 인스턴스 (fit_transform/transform 인터페이스 공통)
    """
    emb_key = embedding_model or "rdkit"
    blocks_key = _normalize_blocks(other_blocks)
    cache_key = (emb_key, pooling_method, use_pca, pca_dim if use_pca else None, blocks_key)

    if cache_key not in _pipeline_cache:
        kwargs = dict(pooling_method=pooling_method, use_pca=use_pca, pca_dim=pca_dim,
                      other_blocks=list(blocks_key))

        if emb_key == "chemberta":
            from heterogeneity.smile_BERTA_gem_pipe import ChemBERTaMediaPipeline
            _pipeline_cache[cache_key] = ChemBERTaMediaPipeline(**kwargs)
        elif emb_key == "unimol":
            from heterogeneity.smile_UniMol_gem_pipe import UniMolMediaPipeline
            _pipeline_cache[cache_key] = UniMolMediaPipeline(**kwargs)
        else:
            from heterogeneity.smile_gem_pipe import MediaPipeline
            _pipeline_cache[cache_key] = MediaPipeline(**kwargs)

    return _pipeline_cache[cache_key]


def get_pipeline_dim_info(embedding_model: str, pooling_method: str = "mean",
                            use_pca: bool = False, pca_dim: int = 30,
                            other_blocks: list = None, pipeline=None) -> dict:
    """
    embedding_model/pooling_method/use_pca/other_blocks 조합에 맞는
    {embedding, metal_physchem, log_conc, gem, pooling_method, pooled_dim, use_pca, total} 반환.
    """
    from heterogeneity.base_pipeline import METAL_PHYSCHEM_DIM, GEM_DIM

    if pipeline is None:
        pipeline = get_pipeline(embedding_model, pooling_method=pooling_method,
                                  use_pca=use_pca, pca_dim=pca_dim, other_blocks=other_blocks)

    return {
        "embedding"      : pipeline._emb_dim,
        "metal_physchem" : METAL_PHYSCHEM_DIM if "metal_physchem" in pipeline.other_blocks else 0,
        "log_conc"       : 1 if "log_conc" in pipeline.other_blocks else 0,
        "gem"            : GEM_DIM if "gem" in pipeline.other_blocks else 0,
        "pooling_method" : pipeline.pooling_method,
        "other_blocks"   : pipeline.other_blocks,
        "pooled_dim"     : pipeline.pooled_dim,
        "use_pca"        : pipeline.use_pca,
        "total"          : pipeline.vector_dim,
    }


def get_all_pipeline_dims() -> dict:
    """
    rdkit/chemberta × mean/multi_stat 조합의 dim 정보를 한 번에 반환.
    (기본 other_blocks=전체 포함 기준. 프론트 카드 하단 표시는 이 기본값으로 보여주고,
     실제 학습 시에는 사용자가 고른 other_blocks로 다시 계산됨.)
    """
    dims = {}
    for emb_key in ("rdkit", "chemberta"):
        dims[emb_key] = {}
        for pooling in ("mean", "multi_stat"):
            dims[emb_key][pooling] = get_pipeline_dim_info(emb_key, pooling_method=pooling)
    return dims