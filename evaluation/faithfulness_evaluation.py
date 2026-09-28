import json

from backend.embedding import EmbeddingModel
from backend.generator import AnswerGenerator
from backend.retrieval import Retriever
from backend.vector_store import VectorStore


INDEX_PATH = "data/processed/vector_store.faiss"
METADATA_PATH = "data/processed/vector_metadata.pkl"

EVALUATOR_MODEL = "gemini-3.7-flash"


EVALUATION_QUERIES = [
    {
        "id": "F1",
        "question": "What is the main objective of this project?",
    },
    {
        "id": "F2",
        "question": "What tools and technologies are used?",
    },
    {
        "id": "F3",
        "question": "What methodology is proposed?",
    },
    {
        "id": "F4",
        "question": "What challenges affect object detection in autonomous vehicles?",
    },
]


def load_retriever() -> Retriever:
    """Load the persisted production retrieval system."""

    vector_store = VectorStore.load(
        INDEX_PATH,
        METADATA_PATH,
    )

    embedding_model = EmbeddingModel()

    return Retriever(
        embedding_model,
        vector_store,
    )


def build_evidence_context(
    retrieved_chunks: list[dict],
) -> str:
    """Build numbered evidence from retrieved chunks."""

    context_parts = []

    for index, result in enumerate(
        retrieved_chunks,
        start=1,
    ):
        metadata = result["metadata"]

        context_parts.append(
            f"[Evidence {index}]\n"
            f"Source: {metadata['source']}\n"
            f"Page: {metadata['page_number']}\n"
            f"Content:\n{metadata['text']}"
        )

    return "\n\n".join(context_parts)


def build_evaluation_prompt(
    question: str,
    answer: str,
    evidence: str,
) -> str:
    """Create a structured LLM-as-a-judge evaluation prompt."""

    return f"""
You are evaluating a Retrieval-Augmented Generation (RAG) system.

Evaluate the generated answer ONLY against the supplied document evidence.

Do not use outside knowledge.

QUESTION:
{question}

GENERATED ANSWER:
{answer}

DOCUMENT EVIDENCE:
{evidence}

Evaluate three dimensions:

1. FAITHFULNESS
Are the factual claims in the answer supported by the supplied evidence?

2. CITATION_CORRECTNESS
Do the source names and page numbers cited in the answer correspond
to evidence that supports the associated claims?

3. ANSWER_RELEVANCE
Does the answer directly and sufficiently address the question?

Scoring:

0.0 = completely incorrect or unsupported
0.5 = partially supported or partially relevant
1.0 = fully supported, correctly cited, and relevant

Return ONLY valid JSON.
Do not use markdown.
Do not include code fences.

Use exactly this structure:

{{
  "faithfulness": 0.0,
  "citation_correctness": 0.0,
  "answer_relevance": 0.0,
  "explanation": "Brief explanation"
}}
"""


def parse_evaluation_response(
    response_text: str,
) -> dict:
    """Parse and validate the evaluator's JSON response."""

    try:
        evaluation = json.loads(
            response_text
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Evaluator returned invalid JSON."
        ) from exc

    required_fields = {
        "faithfulness",
        "citation_correctness",
        "answer_relevance",
        "explanation",
    }

    missing_fields = (
        required_fields
        - evaluation.keys()
    )

    if missing_fields:
        raise RuntimeError(
            "Evaluator response is missing fields: "
            + ", ".join(sorted(missing_fields))
        )

    for field in [
        "faithfulness",
        "citation_correctness",
        "answer_relevance",
    ]:
        value = evaluation[field]

        if not isinstance(
            value,
            (int, float),
        ):
            raise RuntimeError(
                f"Evaluator field '{field}' must be numeric."
            )

        if not 0.0 <= float(value) <= 1.0:
            raise RuntimeError(
                f"Evaluator field '{field}' must be between 0.0 and 1.0."
            )

        evaluation[field] = float(value)

    return evaluation


def evaluate_answer(
    generator: AnswerGenerator,
    question: str,
    answer: str,
    retrieved_chunks: list[dict],
) -> dict:
    """Evaluate one generated answer against retrieved evidence."""

    evidence = build_evidence_context(
        retrieved_chunks
    )

    prompt = build_evaluation_prompt(
        question,
        answer,
        evidence,
    )

    try:
        response = generator.client.models.generate_content(
            model=EVALUATOR_MODEL,
            contents=prompt,
        )

    except Exception as exc:
        raise RuntimeError(
            "The Gemini evaluator request failed. "
            "Please try again later."
        ) from exc

    if not response.text:
        raise RuntimeError(
            "Evaluator returned an empty response."
        )

    return parse_evaluation_response(
        response.text.strip()
    )


def main() -> None:
    """Run answer-level LLM-as-a-judge faithfulness evaluation."""

    print("Loading retrieval system...")

    retriever = load_retriever()
    generator = AnswerGenerator()

    print(
        f"Indexed chunks: "
        f"{retriever.vector_store.index.ntotal}"
    )

    print(
        "\nRunning faithfulness evaluation...\n"
    )

    successful_evaluations = 0
    generation_failures = 0
    evaluator_failures = 0

    faithfulness_scores = []
    citation_scores = []
    relevance_scores = []

    for item in EVALUATION_QUERIES:
        question = item["question"]

        print(
            f"{item['id']}: {question}"
        )

        retrieved_chunks = retriever.retrieve(
            question,
            top_k=5,
        )

        try:
            answer = generator.generate(
                question,
                retrieved_chunks,
            )

        except RuntimeError as exc:
            generation_failures += 1

            print(
                "  Generation status: API_ERROR"
            )

            print(
                f"  Error: {exc}"
            )

            print()
            continue

        print(
            "  Generation status: SUCCESS"
        )

        try:
            evaluation = evaluate_answer(
                generator,
                question,
                answer,
                retrieved_chunks,
            )

            successful_evaluations += 1

            faithfulness_scores.append(
                evaluation["faithfulness"]
            )

            citation_scores.append(
                evaluation["citation_correctness"]
            )

            relevance_scores.append(
                evaluation["answer_relevance"]
            )

            print(
                f"  Faithfulness: "
                f"{evaluation['faithfulness']:.2f}"
            )

            print(
                f"  Citation correctness: "
                f"{evaluation['citation_correctness']:.2f}"
            )

            print(
                f"  Answer relevance: "
                f"{evaluation['answer_relevance']:.2f}"
            )

            print(
                f"  Explanation: "
                f"{evaluation['explanation']}"
            )

        except RuntimeError as exc:
            evaluator_failures += 1

            print(
                "  Evaluator status: API_ERROR"
            )

            print(
                f"  Error: {exc}"
            )

        print()

    print("=" * 70)
    print("Faithfulness Evaluation Summary")
    print("=" * 70)

    print(
        f"Total queries: "
        f"{len(EVALUATION_QUERIES)}"
    )

    print(
        f"Successful evaluations: "
        f"{successful_evaluations}"
    )

    print(
        f"Generation API failures: "
        f"{generation_failures}"
    )

    print(
        f"Evaluator API failures: "
        f"{evaluator_failures}"
    )

    if successful_evaluations:
        print(
            f"Average faithfulness: "
            f"{sum(faithfulness_scores) / len(faithfulness_scores):.4f}"
        )

        print(
            f"Average citation correctness: "
            f"{sum(citation_scores) / len(citation_scores):.4f}"
        )

        print(
            f"Average answer relevance: "
            f"{sum(relevance_scores) / len(relevance_scores):.4f}"
        )

    else:
        print(
            "Average faithfulness: N/A"
        )

        print(
            "Average citation correctness: N/A"
        )

        print(
            "Average answer relevance: N/A"
        )


if __name__ == "__main__":
    main()