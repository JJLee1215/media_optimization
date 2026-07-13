"""
heterogeneity/_registry.py
파이프라인 레지스트리 — RDKit / ChemBERTa / UniMol 인스턴스 생성·캐싱·차원 조회를
한 곳에서 관리.

data_preprocess.py와 routers/train.py, train.py가 모두 이 파일을 통해서만
파이프라인을 생성/조회함. 파이프라인 초기화 로직이 여러 곳에 중복되는 걸 막기 위함.

Usage:
  from heterogeneity._registry import get_pipeline, get_pipeline_dim_info

  pipeline = get_pipeline("chemberta")
  dim_info = get_pipeline_dim_info("chemberta", pipeline)
"""

_pipeline_cache = {}


def get_pipeline(embedding_model: str = "rdkit"):
    """
    embedding_model : "rdkit" | "chemberta" | "unimol"
                      None이면 "rdkit"으로 간주 (하위 호환)
    Returns: 파이프라인 인스턴스 (transform(X, feature_cols) 인터페이스 공통)
    """
    key = embedding_model or "rdkit"
    if key not in _pipeline_cache:
        if key == "chemberta":
            from heterogeneity.smile_BERTA_gem_pipe import ChemBERTaMediaPipeline
            _pipeline_cache[key] = ChemBERTaMediaPipeline()
        elif key == "unimol":
            from heterogeneity.smile_UniMol_gem_pipe import UniMolMediaPipeline
            _pipeline_cache[key] = UniMolMediaPipeline()
        else:
            from heterogeneity.smile_gem_pipe import MediaPipeline
            _pipeline_cache[key] = MediaPipeline()
    return _pipeline_cache[key]


def get_pipeline_dim_info(embedding_model: str, pipeline=None) -> dict:
    """
    embedding_model 종류에 맞는 {embedding, metal_physchem, log_conc, gem, total} 반환.
    pipeline이 None이면 get_pipeline()으로 새로 만들거나 캐시에서 꺼내옴.
    """
    key = embedding_model or "rdkit"

    if key == "chemberta":
        from heterogeneity.smile_BERTA_gem_pipe import METAL_PHYSCHEM_DIM, GEM_DIM
    elif key == "unimol":
        from heterogeneity.smile_UniMol_gem_pipe import METAL_PHYSCHEM_DIM, GEM_DIM
    else:
        from heterogeneity.smile_gem_pipe import METAL_PHYSCHEM_DIM, GEM_DIM

    if pipeline is None:
        pipeline = get_pipeline(key)

    return {
        "embedding"      : pipeline._emb_dim,
        "metal_physchem" : METAL_PHYSCHEM_DIM,
        "log_conc"       : 1,
        "gem"            : GEM_DIM,
        "total"          : pipeline.vector_dim,
    }


def get_all_pipeline_dims() -> dict:
    """rdkit/chemberta/unimol 세 파이프라인의 dim 정보를 한 번에 반환."""
    return {
        key: get_pipeline_dim_info(key)
        for key in ("rdkit", "chemberta", "unimol")
    }