from backend.pdf_processor import extract_text_from_pdf
from backend.text_processor import chunk_pages
from backend.embedding import EmbeddingModel
from backend.vector_store import VectorStore
from backend.retrieval import Retriever


DOCUMENT_PATH = "data/documents/Report.pdf"

EVALUATION_QUERIES = [
    # Objectives
    {
        "id": "Q1",
        "question": "What is the main objective of this project?",
        "relevant_pages": {12},
    },
    {
        "id": "Q2",
        "question": "What does the project aim to achieve?",
        "relevant_pages": {12},
    },
    {
        "id": "Q3",
        "question": "What are the primary goals of the proposed project?",
        "relevant_pages": {12},
    },
    {
        "id": "Q4",
        "question": "Which aspects of autonomous vehicle object detection are being investigated?",
        "relevant_pages": {12},
    },

    # Research gaps
    {
        "id": "Q5",
        "question": "What are the research gaps identified in the project?",
        "relevant_pages": {9},
    },
    {
        "id": "Q6",
        "question": "What problems or gaps does the research attempt to address?",
        "relevant_pages": {9},
    },
    {
        "id": "Q7",
        "question": "What limitations in existing research are identified?",
        "relevant_pages": {9},
    },
    {
        "id": "Q8",
        "question": "What research challenges are highlighted in the project?",
        "relevant_pages": {9},
    },

    # Tools and technologies
    {
        "id": "Q9",
        "question": "What tools and technologies are used?",
        "relevant_pages": {11},
    },
    {
        "id": "Q10",
        "question": "Which technologies are planned for this project?",
        "relevant_pages": {11},
    },
    {
        "id": "Q11",
        "question": "What tools or frameworks are mentioned in the project?",
        "relevant_pages": {11},
    },
    {
        "id": "Q12",
        "question": "Which programming language and deep learning frameworks are used?",
        "relevant_pages": {11},
    },

    # Proposed methodology
    {
        "id": "Q13",
        "question": "What methodology is proposed?",
        "relevant_pages": {13},
    },
    {
        "id": "Q14",
        "question": "How is the proposed system expected to work?",
        "relevant_pages": {13, 14, 15},
    },
    {
        "id": "Q15",
        "question": "What approach is proposed for solving the object detection problem?",
        "relevant_pages": {13, 14, 15},
    },
    {
        "id": "Q16",
        "question": "What are the main steps in the proposed methodology?",
        "relevant_pages": {13},
    },

    # General project questions
    {
        "id": "Q17",
        "question": "Which performance metrics are used to evaluate detection models?",
        "relevant_pages": {12},
    },
    {
        "id": "Q18",
        "question": "Under what environmental conditions is model performance analyzed?",
        "relevant_pages": {12},
    },
    {
        "id": "Q19",
        "question": "What challenges affect object detection in autonomous vehicles?",
        "relevant_pages": {9, 12},
    },
    {
        "id": "Q20",
        "question": "What datasets are mentioned for the project?",
        "relevant_pages": {11},
    },
]


def build_retriever() -> Retriever:
    """Build the retrieval system from the evaluation document."""
    pages = extract_text_from_pdf(DOCUMENT_PATH)

    chunks = chunk_pages(
        pages,
        chunk_size=200,
        chunk_overlap=40,
    )

    embedding_model = EmbeddingModel()

    vectors = embedding_model.encode(
        [chunk["text"] for chunk in chunks]
    )

    vector_store = VectorStore(
        vectors.shape[1]
    )

    vector_store.add(
        vectors,
        chunks,
    )

    return Retriever(
        embedding_model,
        vector_store,
    )


def get_relevant_chunk_ids(
    retriever: Retriever,
    relevant_pages: set[int],
) -> set[str]:
    """
    Convert page-level ground truth into chunk-level ground truth.

    Every chunk belonging to a relevant page is considered relevant.
    """
    return {
        metadata["chunk_id"]
        for metadata in retriever.vector_store.metadata
        if metadata["page_number"] in relevant_pages
    }


def precision_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    """Calculate chunk-level Precision@K."""
    top_k_chunks = retrieved_chunk_ids[:k]

    if not top_k_chunks:
        return 0.0

    relevant_count = sum(
        chunk_id in relevant_chunk_ids
        for chunk_id in top_k_chunks
    )

    return relevant_count / len(top_k_chunks)


def recall_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    """Calculate chunk-level Recall@K."""
    if not relevant_chunk_ids:
        return 0.0

    top_k_chunks = retrieved_chunk_ids[:k]

    relevant_count = sum(
        chunk_id in relevant_chunk_ids
        for chunk_id in top_k_chunks
    )

    return relevant_count / len(relevant_chunk_ids)


def reciprocal_rank(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
) -> float:
    """Calculate chunk-level Reciprocal Rank."""
    for rank, chunk_id in enumerate(
        retrieved_chunk_ids,
        start=1,
    ):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank

    return 0.0

def min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize scores to the [0, 1] range."""
    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:
        return [1.0 for _ in scores]

    return [
        (score - minimum) / (maximum - minimum)
        for score in scores
    ]


def weighted_hybrid_retrieve(
    retriever: Retriever,
    query: str,
    top_k: int,
    dense_weight: float,
) -> list[dict]:
    """
    Retrieve using weighted normalized dense and BM25 scores.

    dense_weight:
        Weight assigned to dense semantic similarity.
        BM25 receives (1 - dense_weight).
    """
    if not 0.0 <= dense_weight <= 1.0:
        raise ValueError(
            "dense_weight must be between 0.0 and 1.0."
        )

    document_count = retriever.vector_store.index.ntotal

    if document_count == 0:
        return []

    dense_results = retriever._dense_retrieve(
        query,
        document_count,
    )

    bm25_results = retriever._bm25_retrieve(
        query,
        document_count,
    )

    dense_scores = min_max_normalize(
        [result["score"] for result in dense_results]
    )

    bm25_scores = min_max_normalize(
        [result["score"] for result in bm25_results]
    )

    dense_by_id = {
        result["metadata"]["chunk_id"]: score
        for result, score in zip(
            dense_results,
            dense_scores,
        )
    }

    bm25_by_id = {
        result["metadata"]["chunk_id"]: score
        for result, score in zip(
            bm25_results,
            bm25_scores,
        )
    }

    metadata_by_id = {
        metadata["chunk_id"]: metadata
        for metadata in retriever.vector_store.metadata
    }

    all_chunk_ids = (
        set(dense_by_id)
        | set(bm25_by_id)
    )

    hybrid_results = []

    for chunk_id in all_chunk_ids:
        dense_score = dense_by_id.get(
            chunk_id,
            0.0,
        )

        bm25_score = bm25_by_id.get(
            chunk_id,
            0.0,
        )

        combined_score = (
            dense_weight * dense_score
            + (1.0 - dense_weight) * bm25_score
        )

        hybrid_results.append(
            {
                "score": combined_score,
                "dense_score": dense_score,
                "bm25_score": bm25_score,
                "metadata": metadata_by_id[chunk_id],
            }
        )

    hybrid_results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return hybrid_results[:top_k]

def evaluate_retriever(
    retriever: Retriever,
    method: str,
    k: int = 5,
) -> dict:
    """Evaluate one retrieval strategy across all benchmark queries."""
    precision_scores = []
    recall_scores = []
    reciprocal_rank_scores = []

    for item in EVALUATION_QUERIES:
        question = item["question"]
        relevant_pages = item["relevant_pages"]

        relevant_chunk_ids = get_relevant_chunk_ids(
            retriever,
            relevant_pages,
        )

        if method == "dense":
            results = retriever._dense_retrieve(
                question,
                retriever.vector_store.index.ntotal,
            )

        elif method == "bm25":
            results = retriever._bm25_retrieve(
                question,
                retriever.vector_store.index.ntotal,
            )

        elif method == "hybrid":
            results = retriever.retrieve(
                question,
                top_k=k,
            )

        elif method == "hybrid_mmr":
            candidates = retriever.retrieve(
                question,
                top_k=max(k * 3, 15),
            )

            results = retriever._mmr_rerank(
                candidates,
                top_k=k,
                lambda_param=0.7,
            )

        else:
            raise ValueError(
                f"Unknown retrieval method: {method}"
            )

        retrieved_chunk_ids = [
            result["metadata"]["chunk_id"]
            for result in results
        ]

        precision_scores.append(
            precision_at_k(
                retrieved_chunk_ids,
                relevant_chunk_ids,
                k,
            )
        )

        recall_scores.append(
            recall_at_k(
                retrieved_chunk_ids,
                relevant_chunk_ids,
                k,
            )
        )

        reciprocal_rank_scores.append(
            reciprocal_rank(
                retrieved_chunk_ids,
                relevant_chunk_ids,
            )
        )

    return {
        "method": method,
        "precision_at_5": sum(precision_scores)
        / len(precision_scores),
        "recall_at_5": sum(recall_scores)
        / len(recall_scores),
        "mrr": sum(reciprocal_rank_scores)
        / len(reciprocal_rank_scores),
    }

def evaluate_weighted_hybrid(
    retriever: Retriever,
    dense_weight: float,
    k: int = 5,
) -> dict:
    """Evaluate weighted dense + BM25 retrieval."""
    precision_scores = []
    recall_scores = []
    reciprocal_rank_scores = []

    for item in EVALUATION_QUERIES:
        question = item["question"]
        relevant_pages = item["relevant_pages"]

        relevant_chunk_ids = get_relevant_chunk_ids(
            retriever,
            relevant_pages,
        )

        results = weighted_hybrid_retrieve(
            retriever,
            question,
            top_k=k,
            dense_weight=dense_weight,
        )

        retrieved_chunk_ids = [
            result["metadata"]["chunk_id"]
            for result in results
        ]

        precision_scores.append(
            precision_at_k(
                retrieved_chunk_ids,
                relevant_chunk_ids,
                k,
            )
        )

        recall_scores.append(
            recall_at_k(
                retrieved_chunk_ids,
                relevant_chunk_ids,
                k,
            )
        )

        reciprocal_rank_scores.append(
            reciprocal_rank(
                retrieved_chunk_ids,
                relevant_chunk_ids,
            )
        )

    return {
        "dense_weight": dense_weight,
        "bm25_weight": 1.0 - dense_weight,
        "precision_at_5": (
            sum(precision_scores)
            / len(precision_scores)
        ),
        "recall_at_5": (
            sum(recall_scores)
            / len(recall_scores)
        ),
        "mrr": (
            sum(reciprocal_rank_scores)
            / len(reciprocal_rank_scores)
        ),
    }

def print_weighted_hybrid_ranking(
    retriever: Retriever,
    dense_weight: float = 0.25,
    k: int = 5,
) -> None:
    """Print the first relevant chunk rank for weighted hybrid retrieval."""
    print(
        f"\nWeighted Hybrid Per-query Ranking "
        f"(Dense={dense_weight:.2f}, "
        f"BM25={1.0 - dense_weight:.2f}):"
    )
    print("-" * 70)

    print(
        f"{'Query':<8}"
        f"{'BM25':<10}"
        f"{'RRF':<10}"
        f"{'Weighted':<12}"
    )

    for item in EVALUATION_QUERIES:
        question = item["question"]
        relevant_pages = item["relevant_pages"]

        relevant_chunk_ids = get_relevant_chunk_ids(
            retriever,
            relevant_pages,
        )

        bm25_results = retriever._bm25_retrieve(
            question,
            retriever.vector_store.index.ntotal,
        )

        rrf_results = retriever.retrieve(
            question,
            top_k=k,
        )

        weighted_results = weighted_hybrid_retrieve(
            retriever,
            question,
            top_k=k,
            dense_weight=dense_weight,
        )

        def first_relevant_rank(results):
            for rank, result in enumerate(
                results[:k],
                start=1,
            ):
                if (
                    result["metadata"]["chunk_id"]
                    in relevant_chunk_ids
                ):
                    return rank
            return None

        bm25_rank = first_relevant_rank(
            bm25_results
        )

        rrf_rank = first_relevant_rank(
            rrf_results
        )

        weighted_rank = first_relevant_rank(
            weighted_results
        )

        print(
            f"{item['id']:<8}"
            f"{str(bm25_rank or '-'): <10}"
            f"{str(rrf_rank or '-'): <10}"
            f"{str(weighted_rank or '-'): <12}"
        )

def print_per_query_ranking(
    retriever: Retriever,
    k: int = 5,
) -> None:
    """Print the first relevant chunk rank for each retrieval method."""

    methods = [
        "dense",
        "bm25",
        "hybrid",
        "hybrid_mmr",
    ]

    print("\nPer-query first relevant chunk rank:")
    print("-" * 75)

    print(
        f"{'Query':<8}"
        f"{'Dense':<10}"
        f"{'BM25':<10}"
        f"{'Hybrid':<10}"
        f"{'MMR':<10}"
    )

    for item in EVALUATION_QUERIES:
        question = item["question"]
        relevant_pages = item["relevant_pages"]

        relevant_chunk_ids = get_relevant_chunk_ids(
            retriever,
            relevant_pages,
        )

        ranks = {}

        for method in methods:

            if method == "dense":
                results = retriever._dense_retrieve(
                    question,
                    retriever.vector_store.index.ntotal,
                )

            elif method == "bm25":
                results = retriever._bm25_retrieve(
                    question,
                    retriever.vector_store.index.ntotal,
                )

            elif method == "hybrid":
                results = retriever.retrieve(
                    question,
                    top_k=k,
                )

                # Production retrieve() now includes MMR,
                # so obtain the pure hybrid ranking separately.
                dense_results = retriever._dense_retrieve(
                    question,
                    retriever.vector_store.index.ntotal,
                )

                bm25_results = retriever._bm25_retrieve(
                    question,
                    retriever.vector_store.index.ntotal,
                )

                results = retriever._weighted_hybrid_retrieve(
                    dense_results,
                    bm25_results,
                    k,
                )

            elif method == "hybrid_mmr":
                candidates = retriever._weighted_hybrid_retrieve(
                    retriever._dense_retrieve(
                        question,
                        min(
                            max(k * 3, 15),
                            retriever.vector_store.index.ntotal,
                        ),
                    ),
                    retriever._bm25_retrieve(
                        question,
                        min(
                            max(k * 3, 15),
                            retriever.vector_store.index.ntotal,
                        ),
                    ),
                    min(
                        max(k * 3, 15),
                        retriever.vector_store.index.ntotal,
                    ),
                )

                results = retriever._mmr_rerank(
                    candidates,
                    top_k=k,
                    lambda_param=0.7,
                )

            else:
                raise ValueError(
                    f"Unknown retrieval method: {method}"
                )

            first_relevant_rank = None

            for rank, result in enumerate(
                results[:k],
                start=1,
            ):
                chunk_id = result["metadata"]["chunk_id"]

                if chunk_id in relevant_chunk_ids:
                    first_relevant_rank = rank
                    break

            ranks[method] = (
                str(first_relevant_rank)
                if first_relevant_rank is not None
                else "-"
            )

        print(
            f"{item['id']:<8}"
            f"{ranks['dense']:<10}"
            f"{ranks['bm25']:<10}"
            f"{ranks['hybrid']:<10}"
            f"{ranks['hybrid_mmr']:<10}"
        )


def print_failed_query_details(
    retriever: Retriever,
    k: int = 5,
) -> None:
    """Print retrieved chunks for selected diagnostic queries."""
    diagnostic_queries = {
        "Q9",
        "Q14",
        "Q15",
    }

    print("\nDiagnostic retrieval details:")
    print("-" * 80)

    for item in EVALUATION_QUERIES:
        if item["id"] not in diagnostic_queries:
            continue

        question = item["question"]
        relevant_pages = item["relevant_pages"]

        relevant_chunk_ids = get_relevant_chunk_ids(
            retriever,
            relevant_pages,
        )

        print(f"\n{item['id']}: {question}")
        print(
            f"Expected pages: "
            f"{sorted(relevant_pages)}"
        )
        print(
            f"Relevant chunks: "
            f"{sorted(relevant_chunk_ids)}"
        )

        dense_results = retriever._dense_retrieve(
            question,
            retriever.vector_store.index.ntotal,
        )

        bm25_results = retriever._bm25_retrieve(
            question,
            retriever.vector_store.index.ntotal,
        )

        hybrid_results = retriever.retrieve(
            question,
            top_k=k,
        )

        for method, results in [
            ("Dense", dense_results[:k]),
            ("BM25", bm25_results[:k]),
            ("Hybrid", hybrid_results[:k]),
        ]:
            print(f"\n{method}:")

            for rank, result in enumerate(
                results,
                start=1,
            ):
                metadata = result["metadata"]

                relevance = (
                    "RELEVANT"
                    if metadata["chunk_id"]
                    in relevant_chunk_ids
                    else "not relevant"
                )

                preview = (
                    metadata["text"][:120]
                    .replace("\n", " ")
                )

                print(
                    f"  Rank {rank}: "
                    f"{metadata['chunk_id']} | "
                    f"Page {metadata['page_number']} | "
                    f"Score {result['score']:.4f} | "
                    f"{relevance} | "
                    f"{preview}..."
                )

def evaluate_mmr_lambda(
    retriever: Retriever,
    lambda_param: float,
    k: int = 5,
) -> dict:
    """Evaluate hybrid retrieval followed by MMR reranking."""
    precision_scores = []
    recall_scores = []
    reciprocal_rank_scores = []

    for item in EVALUATION_QUERIES:
        question = item["question"]
        relevant_pages = item["relevant_pages"]

        relevant_chunk_ids = get_relevant_chunk_ids(
            retriever,
            relevant_pages,
        )

        candidates = retriever.retrieve(
            question,
            top_k=max(k * 3, 15),
        )

        results = retriever._mmr_rerank(
            candidates,
            top_k=k,
            lambda_param=lambda_param,
        )

        retrieved_chunk_ids = [
            result["metadata"]["chunk_id"]
            for result in results
        ]

        precision_scores.append(
            precision_at_k(
                retrieved_chunk_ids,
                relevant_chunk_ids,
                k,
            )
        )

        recall_scores.append(
            recall_at_k(
                retrieved_chunk_ids,
                relevant_chunk_ids,
                k,
            )
        )

        reciprocal_rank_scores.append(
            reciprocal_rank(
                retrieved_chunk_ids,
                relevant_chunk_ids,
            )
        )

    return {
        "lambda": lambda_param,
        "precision_at_5": sum(precision_scores)
        / len(precision_scores),
        "recall_at_5": sum(recall_scores)
        / len(recall_scores),
        "mrr": sum(reciprocal_rank_scores)
        / len(reciprocal_rank_scores),
    }

def main() -> None:
    """Run retrieval evaluation."""
    print("Building retrieval system...")

    retriever = build_retriever()

    print("Running retrieval evaluation...\n")

    methods = [
        "dense",
        "bm25",
        "hybrid",
        "hybrid_mmr",
    ]

    results = [
        evaluate_retriever(
            retriever,
            method,
            k=5,
        )
        for method in methods
    ]

    print(
        "Baseline Retrieval Evaluation:"
    )

    print(
        f"{'Method':<12}"
        f"{'Precision@5':<15}"
        f"{'Recall@5':<12}"
        f"{'MRR':<10}"
    )

    print("-" * 49)

    for result in results:
        print(
            f"{result['method']:<12}"
            f"{result['precision_at_5']:<15.4f}"
            f"{result['recall_at_5']:<12.4f}"
            f"{result['mrr']:.4f}"
        )

    print_per_query_ranking(
        retriever,
        k=5,
    )

    print_failed_query_details(
        retriever,
        k=5,
    )

    print(
        "\nWeighted Hybrid Retrieval Experiment:"
    )

    print("-" * 60)

    weights = [
        0.25,
        0.50,
        0.75,
    ]

    print(
        f"{'Dense Weight':<15}"
        f"{'BM25 Weight':<15}"
        f"{'Precision@5':<15}"
        f"{'Recall@5':<15}"
        f"{'MRR':<10}"
    )

    print("-" * 70)

    for dense_weight in weights:
        result = evaluate_weighted_hybrid(
            retriever,
            dense_weight=dense_weight,
            k=5,
        )

        print(
            f"{result['dense_weight']:<15.2f}"
            f"{result['bm25_weight']:<15.2f}"
            f"{result['precision_at_5']:<15.4f}"
            f"{result['recall_at_5']:<15.4f}"
            f"{result['mrr']:.4f}"
        )

    print_weighted_hybrid_ranking(
        retriever,
        dense_weight=0.25,
        k=5,
    )

    print("\nMMR Lambda Experiment:")
    print("-" * 60)
    print(
        f"{'Lambda':<10}"
        f"{'Precision@5':<15}"
        f"{'Recall@5':<15}"
        f"{'MRR':<10}"
    )

    print("-" * 50)

    for lambda_param in [0.5, 0.7, 0.9]:
        result = evaluate_mmr_lambda(
            retriever,
            lambda_param=lambda_param,
            k=5,
        )

        print(
            f"{result['lambda']:<10.1f}"
            f"{result['precision_at_5']:<15.4f}"
            f"{result['recall_at_5']:<15.4f}"
            f"{result['mrr']:.4f}"
        )


if __name__ == "__main__":
    main()