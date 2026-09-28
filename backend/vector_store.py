from pathlib import Path
import pickle

import faiss
import numpy as np


class VectorStore:
    """FAISS-based vector store with persistent metadata."""

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0.")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []

    def add(
        self,
        vectors: np.ndarray,
        metadata: list[dict],
    ) -> None:
        """Add normalized vectors and their metadata to the index."""
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array.")

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vectors with dimension {self.dimension}, "
                f"got {vectors.shape[1]}."
            )

        if len(vectors) != len(metadata):
            raise ValueError(
                "Number of vectors must match number of metadata records."
            )

        self.index.add(
            vectors.astype(np.float32)
        )

        self.metadata.extend(metadata)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """Return the top-k most similar records."""
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if (
            query_vector.ndim != 2
            or query_vector.shape[1] != self.dimension
        ):
            raise ValueError(
                f"query_vector must have shape (n, {self.dimension})."
            )

        if self.index.ntotal == 0:
            return []

        scores, indices = self.index.search(
            query_vector.astype(np.float32),
            min(top_k, self.index.ntotal),
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index == -1:
                continue

            results.append(
                {
                    "score": float(score),
                    "metadata": self.metadata[index],
                }
            )

        return results

    def save(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> None:
        """Persist the FAISS index and metadata to disk."""
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(index_path),
        )

        with metadata_path.open(
            "wb"
        ) as file:
            pickle.dump(
                {
                    "dimension": self.dimension,
                    "metadata": self.metadata,
                },
                file,
            )

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> "VectorStore":
        """Load a persisted FAISS index and metadata."""
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        index = faiss.read_index(
            str(index_path)
        )

        with metadata_path.open(
            "rb"
        ) as file:
            stored_data = pickle.load(file)

        dimension = stored_data["dimension"]
        metadata = stored_data["metadata"]

        if index.d != dimension:
            raise ValueError(
                "FAISS index dimension does not match stored dimension."
            )

        if index.ntotal != len(metadata):
            raise ValueError(
                "FAISS index size does not match metadata size."
            )

        vector_store = cls(
            dimension=dimension
        )

        vector_store.index = index
        vector_store.metadata = metadata

        return vector_store