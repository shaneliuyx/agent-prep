"""rag_hybrid — modular hybrid-RAG building blocks.

Public API:

    from rag_hybrid import (
        # Specs (move from lab-02 model_config — frozen dataclasses)
        EmbedModelSpec, RerankerSpec,
        CollectionSpec, HybridCollectionSpec,
        BGE_M3, BGE_RERANKER_V2_M3, BGE_M3_HYBRID, BGE_M3_HNSW,

        # Encoders (lazy-loaded model wrappers)
        EncoderConfig, HybridEncoder, DenseEncoder,

        # Reranker
        RerankerConfig, CrossEncoderReranker,

        # Ingest
        IngestConfig, Ingestor, chunk_corpus,

        # Retrieve
        RetrieveConfig, Retriever,

        # Fusion
        FusionStrategy, rrf_fuse,

        # Schema
        ensure_collection, detect_collection_mode,

        # Chunking
        char_window_chunks, sentence_window_chunks,
    )

    from rag_hybrid import autoconfig
    cfg = autoconfig.recommend(BGE_M3, BGE_RERANKER_V2_M3)
"""

# Specs
from .models import (  # noqa: F401
    BGE_M3,
    BGE_M3_HNSW,
    BGE_M3_HYBRID,
    BGE_RERANKER_V2_M3,
    FIQA_HYBRID,
    ALL_COLLECTIONS,
    SWEEP_VARIANTS,
    LONG_DOC_PASSAGES,
    CHUNK_SIZES,
    CHUNK_OVERLAPS,
    ChunkSweepCollectionSpec,
    CollectionSpec,
    EmbedModelSpec,
    HybridCollectionSpec,
    RerankerSpec,
)

# Encoders
from .encoder import (  # noqa: F401
    DenseEncoder,
    EncoderConfig,
    HybridEncoder,
)

# Reranker
from .rerank import (  # noqa: F401
    CrossEncoderReranker,
    RerankerConfig,
)

# Ingest
from .ingest import (  # noqa: F401
    IngestConfig,
    Ingestor,
    chunk_corpus,
)

# Retrieve
from .retrieve import (  # noqa: F401
    RetrieveConfig,
    Retriever,
)

# Fusion
from .fusion import (  # noqa: F401
    FusionStrategy,
    RRF_DEFAULT_K,
    rrf_fuse,
)

# Schema
from .schema import (  # noqa: F401
    detect_collection_mode,
    ensure_collection,
)

# Chunking
from .chunking import (  # noqa: F401
    char_window_chunks,
    sentence_window_chunks,
)

__version__ = "0.1.0"
