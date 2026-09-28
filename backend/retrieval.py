import re

import numpy as np

from rank_bm25 import BM25Okapi

from backend.embedding import EmbeddingModel
from backend.vector_store import VectorStore


class Retriever:
    """Hybrid dense + lexical retriever with MMR reranking."""

    DENSE_WEIGHT = 0.25
    BM25_WEIGHT = 0.75
    MMR_LAMBDA = 0.7

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

        self.documents = [
            metadata["text"]
            for metadata in self.vector_store.metadata
        ]

        self.tokenized_documents = [
            self._tokenize(text)
            for text in self.documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

    @staticmethod
    def _stem(word: str) -> str:
        """Apply lightweight normalization for common English word forms."""
        word = word.lower()

        irregular_forms = {
            "objectives": "objective",
            "studies": "study",
            "analyses": "analysis",
            "technologies": "technology",
        }

        if word in irregular_forms:
            return irregular_forms[word]

        if len(word) > 6 and word.endswith("ies"):
            return word[:-3] + "y"

        if len(word) > 7 and word.endswith("ing"):
            return word[:-3]

        if len(word) > 6 and word.endswith("ed"):
            return word[:-2]

        if len(word) > 5 and word.endswith("s"):
            return word[:-1]

        return word

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        """Tokenize and normalize text for BM25."""
        words = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )

        return [
            cls._stem(word)
            for word in words
        ]

    def _dense_retrieve(
        self,
        query: str,
        candidate_k: int,
    ) -> list[dict]:
        """Retrieve candidates using semantic similarity."""
        query_vector = self.embedding_model.encode(
            [query]
        )

        return self.vector_store.search(
            query_vector,
            top_k=candidate_k,
        )

    def _bm25_retrieve(
        self,
        query: str,
        candidate_k: int,
    ) -> list[dict]:
        """Retrieve candidates using BM25 lexical matching."""
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = scores.argsort()[::-1][
            :candidate_k
        ]

        results = []

        for index in ranked_indices:
            results.append(
                {
                    "score": float(scores[index]),
                    "metadata": self.vector_store.metadata[
                        index
                    ],
                }
            )

        return results

    @staticmethod
    def _min_max_normalize(
        scores: list[float],
    ) -> np.ndarray:
        """Normalize scores to the range [0, 1]."""
        values = np.asarray(
            scores,
            dtype=np.float32,
        )

        if len(values) == 0:
            return values

        minimum = values.min()
        maximum = values.max()

        if maximum == minimum:
            return np.ones_like(values)

        return (
            (values - minimum)
            / (maximum - minimum)
        )

    def _weighted_hybrid_retrieve(
        self,
        dense_results: list[dict],
        bm25_results: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Combine dense and BM25 retrieval using weighted score fusion.

        Dense retrieval contributes 25%.
        BM25 retrieval contributes 75%.
        """

        dense_scores = self._min_max_normalize(
            [
                result["score"]
                for result in dense_results
            ]
        )

        bm25_scores = self._min_max_normalize(
            [
                result["score"]
                for result in bm25_results
            ]
        )

        combined_results = {}

        for index, result in enumerate(
            dense_results
        ):
            chunk_id = result["metadata"]["chunk_id"]

            combined_results.setdefault(
                chunk_id,
                {
                    "dense_score": 0.0,
                    "bm25_score": 0.0,
                    "metadata": result["metadata"],
                },
            )

            combined_results[
                chunk_id
            ]["dense_score"] = float(
                dense_scores[index]
            )

        for index, result in enumerate(
            bm25_results
        ):
            chunk_id = result["metadata"]["chunk_id"]

            combined_results.setdefault(
                chunk_id,
                {
                    "dense_score": 0.0,
                    "bm25_score": 0.0,
                    "metadata": result["metadata"],
                },
            )

            combined_results[
                chunk_id
            ]["bm25_score"] = float(
                bm25_scores[index]
            )

        hybrid_results = []

        for result in combined_results.values():
            weighted_score = (
                self.DENSE_WEIGHT
                * result["dense_score"]
                + self.BM25_WEIGHT
                * result["bm25_score"]
            )

            hybrid_results.append(
                {
                    "score": weighted_score,
                    "dense_score": result["dense_score"],
                    "bm25_score": result["bm25_score"],
                    "metadata": result["metadata"],
                }
            )

        # Sort AFTER all hybrid candidates have been created.
        hybrid_results.sort(
            key=lambda result: result["score"],
            reverse=True,
        )

        return hybrid_results[:top_k]

    def _mmr_rerank(
        self,
        candidates: list[dict],
        top_k: int,
        lambda_param: float = MMR_LAMBDA,
    ) -> list[dict]:
        """
        Rerank candidates using Maximal Marginal Relevance.

        MMR balances retrieval relevance with diversity among
        the selected chunks.
        """

        if not candidates:
            return []

        if not 0 < lambda_param <= 1:
            raise ValueError(
                "lambda_param must be greater than 0 "
                "and less than or equal to 1."
            )

        top_k = min(
            top_k,
            len(candidates),
        )

        candidate_texts = [
            candidate["metadata"]["text"]
            for candidate in candidates
        ]

        candidate_embeddings = (
            self.embedding_model.encode(
                candidate_texts
            )
        )

        selected_indices = []
        remaining_indices = list(
            range(len(candidates))
        )

        while (
            remaining_indices
            and len(selected_indices) < top_k
        ):
            best_index = None
            best_score = float("-inf")

            for index in remaining_indices:
                relevance = candidates[index]["score"]

                if selected_indices:
                    diversity_penalty = max(
                        float(
                            np.dot(
                                candidate_embeddings[index],
                                candidate_embeddings[
                                    selected_index
                                ],
                            )
                        )
                        for selected_index in selected_indices
                    )
                else:
                    diversity_penalty = 0.0

                mmr_score = (
                    lambda_param * relevance
                    - (
                        1 - lambda_param
                    )
                    * diversity_penalty
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = index

            selected_indices.append(
                best_index
            )

            remaining_indices.remove(
                best_index
            )

        reranked_results = []

        for index in selected_indices:
            result = candidates[index].copy()

            relevance = candidates[index]["score"]

            if selected_indices.index(index) == 0:
                diversity_penalty = 0.0
            else:
                previously_selected = selected_indices[
                    :selected_indices.index(index)
                ]

                diversity_penalty = max(
                    float(
                        np.dot(
                            candidate_embeddings[index],
                            candidate_embeddings[selected_index],
                        )
                    )
                    for selected_index in previously_selected
                )

            result["mmr_score"] = float(
                lambda_param * relevance
                - (1 - lambda_param) * diversity_penalty
            )

            reranked_results.append(result)

        return reranked_results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve relevant chunks using hybrid retrieval
        followed by MMR reranking.
        """

        if not query.strip():
            raise ValueError(
                "Query must not be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        document_count = (
            self.vector_store.index.ntotal
        )

        if document_count == 0:
            return []

        # Retrieve a larger candidate pool first.
        candidate_k = min(
            max(top_k * 3, 15),
            document_count,
        )

        dense_results = self._dense_retrieve(
            query,
            candidate_k,
        )

        bm25_results = self._bm25_retrieve(
            query,
            candidate_k,
        )

        hybrid_candidates = (
            self._weighted_hybrid_retrieve(
                dense_results,
                bm25_results,
                candidate_k,
            )
        )

        # Apply MMR to balance relevance and diversity.
        return self._mmr_rerank(
            hybrid_candidates,
            top_k=top_k,
            lambda_param=self.MMR_LAMBDA,
        )