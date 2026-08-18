"""Vector retrieval utilities for NewsLens."""

from .base import (
    RetrievalError,
    RetrievalHit,
)
from .catalog import (
    RetrievalCatalog,
)
from .exact import (
    ExactInnerProductRetriever,
)
from .metrics import (
    retrieval_recall_at_k,
)
from .queries import (
    RetrievalQuery,
    build_validation_queries,
)

__all__ = [
    "ExactInnerProductRetriever",
    "RetrievalCatalog",
    "RetrievalError",
    "RetrievalHit",
    "RetrievalQuery",
    "build_validation_queries",
    "retrieval_recall_at_k",
]
