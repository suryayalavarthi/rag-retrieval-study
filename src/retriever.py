"""
Dense retrieval using Contriever + FAISS.

Handles passage encoding, index building, and retrieval.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Dense passage retrieval using Contriever and FAISS."""

    def __init__(
        self,
        model_name: str = "facebook/contriever-msmarco",
        device: str = "cpu",
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self.index = None
        self.passages = []  # Store raw passage texts
        self.passage_ids = []  # Store passage identifiers

    def load_model(self):
        """Load the Contriever model and tokenizer."""
        from transformers import AutoModel, AutoTokenizer
        import torch

        logger.info(f"Loading retriever model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        logger.info("Retriever model loaded successfully")

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into dense vectors."""
        import torch

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling over token embeddings
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

        return embeddings.cpu().numpy().astype("float32")

    def encode_passages(self, passages: list[str], passage_ids: Optional[list[str]] = None) -> np.ndarray:
        """Encode all passages into dense vectors."""
        if self.model is None:
            self.load_model()

        self.passages = passages
        self.passage_ids = passage_ids or [str(i) for i in range(len(passages))]

        all_embeddings = []
        for i in range(0, len(passages), self.batch_size):
            batch = passages[i:i + self.batch_size]
            embeddings = self._encode_batch(batch)
            all_embeddings.append(embeddings)
            if (i // self.batch_size) % 10 == 0:
                logger.info(f"Encoded {i + len(batch)}/{len(passages)} passages")

        return np.vstack(all_embeddings)

    def build_index(self, embeddings: np.ndarray, index_type: str = "faiss_flat"):
        """Build a FAISS index from passage embeddings."""
        import faiss

        dim = embeddings.shape[1]
        logger.info(f"Building FAISS index: type={index_type}, dim={dim}, n={len(embeddings)}")

        if index_type == "faiss_flat":
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vectors)
        elif index_type == "faiss_ivf":
            nlist = min(100, len(embeddings) // 10)
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.nprobe = 10
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        logger.info(f"FAISS index built with {self.index.ntotal} passages")

    def retrieve(self, query: str, top_k: int = 5) -> tuple[list[str], list[float]]:
        """Retrieve top-k passages for a query."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        if self.model is None:
            self.load_model()

        import faiss

        # Encode query
        query_embedding = self._encode_batch([query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        scores = scores[0].tolist()
        indices = indices[0].tolist()

        # Get passages
        retrieved_passages = [self.passages[idx] for idx in indices if idx >= 0]
        retrieved_scores = [scores[i] for i, idx in enumerate(indices) if idx >= 0]

        return retrieved_passages, retrieved_scores

    def batch_retrieve(self, queries: list[str], top_k: int = 5) -> list[tuple[list[str], list[float]]]:
        """Batch retrieve for multiple queries."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        if self.model is None:
            self.load_model()

        import faiss

        results = []
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            query_embeddings = self._encode_batch(batch)
            faiss.normalize_L2(query_embeddings)

            scores, indices = self.index.search(query_embeddings, top_k)

            for j in range(len(batch)):
                passages = [self.passages[idx] for idx in indices[j] if idx >= 0]
                passage_scores = [scores[j][k] for k, idx in enumerate(indices[j]) if idx >= 0]
                results.append((passages, passage_scores))

        return results

    def save_index(self, path: str):
        """Save the FAISS index and passage data."""
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "passages.pkl", "wb") as f:
            pickle.dump({
                "passages": self.passages,
                "passage_ids": self.passage_ids,
            }, f)
        logger.info(f"Index saved to {path}")

    def load_index(self, path: str):
        """Load a saved FAISS index and passage data."""
        import faiss

        path = Path(path)
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "passages.pkl", "rb") as f:
            data = pickle.load(f)
            self.passages = data["passages"]
            self.passage_ids = data["passage_ids"]
        logger.info(f"Index loaded from {path}: {self.index.ntotal} passages")
