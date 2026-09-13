"""Local e5 embeddings (replaces Pinecone's hosted inference for the pgvector backend).

e5 models are trained with role prefixes: documents as ``passage: …`` and questions as
``query: …``. Pinecone applies them server-side; here they are explicit. Omitting them
silently degrades retrieval, so callers never add them themselves.

Vectors are L2-normalized, so cosine similarity == inner product.
"""
from __future__ import annotations

import os

import numpy as np

from .. import config

_model = None


def _device() -> str:
    forced = os.environ.get("MPCR_EMBED_DEVICE")
    if forced:
        return forced
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.LOCAL_EMBED_MODEL, device=_device())
    return _model


def embed_passages(texts: list[str], *, batch_size: int = 16,
                   show_progress: bool = False) -> np.ndarray:
    return model().encode([f"passage: {t}" for t in texts], batch_size=batch_size,
                          normalize_embeddings=True, show_progress_bar=show_progress)


def embed_query(text: str) -> np.ndarray:
    return model().encode([f"query: {text}"], normalize_embeddings=True,
                          show_progress_bar=False)[0]
