from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """Generate semantic embeddings for text."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]):
        """Convert a list of texts into embedding vectors."""
        if not texts:
            raise ValueError("texts must not be empty.")

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )