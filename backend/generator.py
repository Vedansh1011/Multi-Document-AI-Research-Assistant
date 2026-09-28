import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types


MODEL_NAMES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
]

MAX_RETRIES_PER_MODEL = 2


class AnswerGenerator:
    """Generate grounded answers from retrieved document context."""

    def __init__(self):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(api_key=api_key)

    def _generate_with_model(self, model_name: str, prompt: str) -> str:
        """Generate an answer using one Gemini model with bounded retries."""

        last_error = None

        for attempt in range(MAX_RETRIES_PER_MODEL):
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level="low"
                        )
                    ),
                )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return response.text.strip()

            except errors.ServerError as exc:
                last_error = exc

                print(
                    f"Gemini 503 from {model_name} "
                    f"(attempt {attempt + 1}/{MAX_RETRIES_PER_MODEL})."
                )

                if attempt < MAX_RETRIES_PER_MODEL - 1:
                    time.sleep(2 ** attempt)

            except errors.ClientError as exc:
                print("\n--- Gemini Client Error ---")
                print(f"Type: {type(exc).__name__}")
                print(f"Code: {getattr(exc, 'code', 'unknown')}")
                print(f"Status: {getattr(exc, 'status', 'unknown')}")
                print(f"Message: {exc}")
                print("---------------------------\n")

                if getattr(exc, "code", None) == 429:
                    raise RuntimeError(
                        "Gemini API quota has been exhausted. "
                        "Please wait for the quota to reset."
                    ) from exc

                raise RuntimeError(
                    "The Gemini API request failed."
                ) from exc

            except errors.APIError as exc:
                print("\n--- Gemini API Error ---")
                print(f"Type: {type(exc).__name__}")
                print(f"Code: {getattr(exc, 'code', 'unknown')}")
                print(f"Status: {getattr(exc, 'status', 'unknown')}")
                print(f"Message: {exc}")
                print("------------------------\n")

                raise RuntimeError(
                    "The Gemini API request failed."
                ) from exc

        raise RuntimeError(
            f"Model {model_name} remained unavailable."
        ) from last_error

    def generate(
        self,
        question: str,
        retrieved_chunks: list[dict],
    ) -> str:
        """Generate an answer using only the retrieved document context."""

        if not question.strip():
            raise ValueError("Question must not be empty.")

        if not retrieved_chunks:
            return (
                "I don't have enough information in the provided "
                "documents to answer this question."
            )

        context_parts = []

        for index, result in enumerate(
            retrieved_chunks,
            start=1,
        ):
            metadata = result["metadata"]

            context_parts.append(
                f"[Context {index}]\n"
                f"Source: {metadata['source']}\n"
                f"Page: {metadata['page_number']}\n"
                f"Content:\n{metadata['text']}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a research assistant answering questions from uploaded documents.

Answer the user's question using ONLY the provided document context.

Rules:

1. Do not use outside knowledge.

2. Do not invent facts that are not supported by the context.

3. If the context does not contain enough information, say:

   "I don't have enough information in the provided documents to answer this question."

4. Give a clear and concise answer.

5. When making a factual claim, cite the supporting source using this format:

   [Source: filename, Page: X]

6. Use only the source names and page numbers provided in the context.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

        last_error = None

        for model_name in MODEL_NAMES:
            try:
                print(f"Trying Gemini model: {model_name}")

                answer = self._generate_with_model(
                    model_name=model_name,
                    prompt=prompt,
                )

                print(f"Gemini model succeeded: {model_name}")

                return answer

            except RuntimeError as exc:
                last_error = exc

                print(
                    f"Model {model_name} unavailable. "
                    "Trying the next model..."
                )

        raise RuntimeError(
            "All configured Gemini models are temporarily unavailable. "
            "Please try again shortly."
        ) from last_error