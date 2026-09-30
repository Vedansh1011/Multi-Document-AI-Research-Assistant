import os

import numpy as np
from google import genai
from google.genai import types


MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768


class EmbeddingModel:
    """Generate semantic embeddings using the Gemini Embedding API."""

    def __init__(self):
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable is required.")

        self.client = genai.Client()
        self.model_name = MODEL_NAME

    def encode(self, texts: list[str]) -> np.ndarray:
        """Convert a list of texts into normalized embedding vectors."""

        if not texts:
            raise ValueError("texts must not be empty.")

        result = self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBEDDING_DIMENSION,
            ),
        )

        embeddings = np.array(
            [embedding.values for embedding in result.embeddings],
            dtype=np.float32,
        )

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        embeddings = embeddings / np.maximum(norms, 1e-12)

        return embeddings