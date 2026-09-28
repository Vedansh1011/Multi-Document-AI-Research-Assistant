from backend.embedding import EmbeddingModel
from backend.generator import AnswerGenerator
from backend.retrieval import Retriever
from backend.vector_store import VectorStore


INDEX_PATH = "data/processed/vector_store.faiss"
METADATA_PATH = "data/processed/vector_metadata.pkl"


GENERATION_QUERIES = [
    {
        "id": "G1",
        "question": "What is the main objective of this project?",
        "expected_pages": {12},
        "answerable": True,
    },
    {
        "id": "G2",
        "question": "What tools and technologies are used?",
        "expected_pages": {11},
        "answerable": True,
    },
    {
        "id": "G3",
        "question": "What research gaps are identified in the project?",
        "expected_pages": {9},
        "answerable": True,
    },
    {
        "id": "G4",
        "question": "What methodology is proposed?",
        "expected_pages": {13, 14, 15},
        "answerable": True,
    },
    {
        "id": "G5",
        "question": "What challenges affect object detection in autonomous vehicles?",
        "expected_pages": {5, 9, 12},
        "answerable": True,
    },
    {
        "id": "G6",
        "question": "What is the training accuracy of the final YOLO model?",
        "expected_pages": set(),
        "answerable": False,
    },
    {
        "id": "G7",
        "question": "What is the exact inference latency of the deployed system?",
        "expected_pages": set(),
        "answerable": False,
    },
    {
        "id": "G8",
        "question": "What is the final production mAP achieved by the model?",
        "expected_pages": set(),
        "answerable": False,
    },
]


def load_retriever() -> Retriever:
    """Load the persisted production retrieval system."""

    store = VectorStore.load(
        INDEX_PATH,
        METADATA_PATH,
    )

    embedding_model = EmbeddingModel()

    return Retriever(
        embedding_model,
        store,
    )


def evaluate_citation_presence(
    answer: str,
    answerable: bool,
) -> bool:
    """
    Check whether an answer contains source/page citations.

    Citations are required for answerable questions.
    Citation presence is not evaluated for unanswerable questions.
    """

    if not answerable:
        return True

    return (
        "[Source:" in answer
        and "Page:" in answer
    )


def evaluate_expected_pages(
    retrieved_chunks: list[dict],
    expected_pages: set[int],
    answerable: bool,
) -> bool:
    """
    Check whether retrieved context contains expected source pages.

    For answerable questions, at least one expected page must
    appear in the retrieved context.

    For unanswerable questions, retrieval coverage is not applicable.
    """

    if not answerable:
        return True

    if not expected_pages:
        return False

    retrieved_pages = {
        result["metadata"]["page_number"]
        for result in retrieved_chunks
    }

    return bool(
        retrieved_pages.intersection(
            expected_pages
        )
    )


def evaluate_abstention(
    answer: str,
    answerable: bool,
) -> bool:
    """
    Check whether the model abstains when the answer
    is expected to be unavailable.
    """

    if answerable:
        return True

    abstention_phrases = [
        "I don't have enough information",
        "not provided in the documents",
        "not available in the provided documents",
        "cannot be determined from the provided documents",
        "does not provide",
        "not mentioned in the provided documents",
    ]

    normalized_answer = answer.lower()

    return any(
        phrase.lower() in normalized_answer
        for phrase in abstention_phrases
    )


def main() -> None:
    """Run generation evaluation."""

    print("Loading retrieval system...")

    retriever = load_retriever()
    generator = AnswerGenerator()

    print(
        f"Indexed chunks: "
        f"{retriever.vector_store.index.ntotal}"
    )

    print(
        "\nRunning generation evaluation...\n"
    )

    citation_results = []
    abstention_results = []
    retrieval_results = []

    successful_generations = 0
    api_failures = 0

    answerable_count = 0
    unanswerable_count = 0

    for item in GENERATION_QUERIES:
        question = item["question"]
        expected_pages = item["expected_pages"]
        answerable = item["answerable"]

        if answerable:
            answerable_count += 1
        else:
            unanswerable_count += 1

        print(
            f"{item['id']}: {question}"
        )

        retrieved_chunks = retriever.retrieve(
            question,
            top_k=5,
        )

        retrieval_ok = evaluate_expected_pages(
            retrieved_chunks,
            expected_pages,
            answerable,
        )

        if answerable:
            retrieval_results.append(
                retrieval_ok
            )

        try:
            answer = generator.generate(
                question,
                retrieved_chunks,
            )

            successful_generations += 1

            citation_ok = evaluate_citation_presence(
                answer,
                answerable,
            )

            abstention_ok = evaluate_abstention(
                answer,
                answerable,
            )

            if answerable:
                citation_results.append(
                    citation_ok
                )
            else:
                abstention_results.append(
                    abstention_ok
                )

            print(
                f"  Retrieved expected page: "
                f"{'YES' if retrieval_ok else 'NO'}"
            )

            print(
                "  Generation status: SUCCESS"
            )

            if answerable:
                print(
                    f"  Citation check: "
                    f"{'PASS' if citation_ok else 'FAIL'}"
                )
            else:
                print(
                    f"  Grounding/abstention check: "
                    f"{'PASS' if abstention_ok else 'FAIL'}"
                )

            print(
                f"  Answer preview: "
                f"{answer[:200].replace(chr(10), ' ')}..."
            )

        except RuntimeError as exc:
            api_failures += 1

            print(
                f"  Retrieved expected page: "
                f"{'YES' if retrieval_ok else 'NO'}"
            )

            print(
                "  Generation status: API_ERROR"
            )

            print(
                f"  Generation error: {exc}"
            )

        print()

    retrieval_accuracy = (
        sum(retrieval_results)
        / len(retrieval_results)
        if retrieval_results
        else 0.0
    )

    citation_accuracy = (
        sum(citation_results)
        / len(citation_results)
        if citation_results
        else 0.0
    )

    abstention_accuracy = (
        sum(abstention_results)
        / len(abstention_results)
        if abstention_results
        else 0.0
    )

    print("=" * 65)
    print("Generation Evaluation Summary")
    print("=" * 65)

    print(
        f"Total evaluation queries: "
        f"{len(GENERATION_QUERIES)}"
    )

    print(
        f"Successful generations: "
        f"{successful_generations}"
    )

    print(
        f"API failures: "
        f"{api_failures}"
    )

    print(
        f"Answerable queries: "
        f"{answerable_count}"
    )

    print(
        f"Unanswerable queries: "
        f"{unanswerable_count}"
    )

    if retrieval_results:
        print(
            f"Expected-page retrieval coverage "
            f"(answerable queries): "
            f"{retrieval_accuracy:.4f}"
        )
    else:
        print(
            "Expected-page retrieval coverage "
            "(answerable queries): N/A"
        )

    if citation_results:
        print(
            f"Citation presence accuracy "
            f"(successful answerable generations): "
            f"{citation_accuracy:.4f}"
        )
    else:
        print(
            "Citation presence accuracy "
            "(successful answerable generations): N/A"
        )

    if abstention_results:
        print(
            f"Grounding/abstention accuracy "
            f"(successful unanswerable generations): "
            f"{abstention_accuracy:.4f}"
        )
    else:
        print(
            "Grounding/abstention accuracy "
            "(successful unanswerable generations): N/A"
        )


if __name__ == "__main__":
    main()