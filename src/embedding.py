"""SANAD embedding utilities for Qwen3-Embedding-0.6B and BGE-M3."""

from sentence_transformers import SentenceTransformer
import numpy as np
import torch

QWEN_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BGE_MODEL_NAME = "BAAI/bge-m3"

QWEN_QUERY_INSTRUCTION = (
    "Instruct: Retrieve the most relevant Saudi legal article "
    "for the given Arabic legal question.\n"
    "Query: "
)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embedding_model(model_name: str, max_seq_length: int | None = None):
    model = SentenceTransformer(model_name, device=get_device())
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
    return model


def encode_documents(model, documents: list[str], batch_size: int = 16) -> np.ndarray:
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    return np.asarray(embeddings, dtype="float32")


def prepare_query(query: str, model_name: str) -> str:
    if "qwen3-embedding" in model_name.lower():
        return QWEN_QUERY_INSTRUCTION + str(query)
    return str(query)


def encode_query(model, query: str, model_name: str) -> np.ndarray:
    prepared_query = prepare_query(query, model_name)
    embedding = model.encode([prepared_query], normalize_embeddings=True)
    return np.asarray(embedding, dtype="float32")
