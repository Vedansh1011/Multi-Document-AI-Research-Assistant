import os
import time
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types


MODEL_NAMES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]

MAX_RETRIES_PER_MODEL = 2
FORMULA_PAGE_BATCH_SIZE = 4

DOCUMENTS_DIR = Path("data/documents")


class AnswerGenerator:
    """Generate grounded answers from uploaded document context."""

    def __init__(self):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        self.client = genai.Client(
            api_key=api_key
        )
        self.exhausted_models = set()

    def _generate_with_model(
        self,
        model_name: str,
        prompt: str,
        page_images: list[bytes],
    ) -> str:
        """Generate an answer using Gemini with bounded retries."""

        last_error = None

        contents = [prompt]

        for image_bytes in page_images:
            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                )
            )

        for attempt in range(MAX_RETRIES_PER_MODEL):
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
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
                    f"(attempt {attempt + 1}/"
                    f"{MAX_RETRIES_PER_MODEL})."
                )

                if attempt < MAX_RETRIES_PER_MODEL - 1:
                    time.sleep(2 ** attempt)

            except errors.ClientError as exc:
                code = getattr(exc, "code", None)

                print(
                    f"Gemini client error from {model_name}: "
                    f"code={code}, message={exc}"
                )

                if code == 429:
                    self.exhausted_models.add(model_name)

                    raise RuntimeError(
                        f"Gemini model {model_name} returned "
                        "429 RESOURCE_EXHAUSTED and has been temporarily "
                        "disabled for this server session."
                    ) from exc

                raise RuntimeError(
                    "The Gemini API request failed."
                ) from exc

            except errors.APIError as exc:
                print(
                    f"Gemini API error from {model_name}: "
                    f"code={getattr(exc, 'code', 'unknown')}, "
                    f"message={exc}"
                )

                raise RuntimeError(
                    "The Gemini API request failed."
                ) from exc

        raise RuntimeError(
            f"Model {model_name} remained unavailable."
        ) from last_error

    def _load_page_images(
        self,
        page_records: list[dict],
    ) -> list[bytes]:
        """Render original PDF pages as PNG images."""

        page_images = []

        for page_record in page_records:
            source = page_record["source"]
            page_number = page_record["page_number"]

            pdf_path = DOCUMENTS_DIR / Path(source).name

            if not pdf_path.exists():
                print(
                    f"PDF not found: {pdf_path}"
                )
                continue

            try:
                document = pymupdf.open(pdf_path)

                try:
                    page_index = int(page_number) - 1

                    if (
                        page_index < 0
                        or page_index >= len(document)
                    ):
                        continue

                    page = document[page_index]

                    matrix = pymupdf.Matrix(
                        1.5,
                        1.5,
                    )

                    pixmap = page.get_pixmap(
                        matrix=matrix,
                        alpha=False,
                    )

                    page_images.append(
                        pixmap.tobytes("png")
                    )

                finally:
                    document.close()

            except Exception as exc:
                print(
                    f"Unable to render PDF page "
                    f"{source} / page {page_number}: {exc}"
                )

        return page_images

    def _is_formula_question(
        self,
        question: str,
    ) -> bool:
        """Detect questions requiring comprehensive formula extraction."""

        keywords = [
            "formula",
            "formulas",
            "equation",
            "equations",
            "mathematical",
            "mathematics",
            "derive",
            "derivation",
            "loss function",
            "loss functions",
            "attention mechanism",
            "optimization",
            "self-supervised",
            "adversarial training",
        ]

        question_lower = question.lower()

        return any(
            keyword in question_lower
            for keyword in keywords
        )

    def _get_formula_candidate_pages(self) -> list[dict]:
        """
        Identify pages that are likely to contain mathematical formulas.

        Candidate pages are selected from extracted PDF text using formula
        markers and section terminology. The original PDF page image is
        still used by Gemini for the final mathematical reconstruction.
        """

        candidate_pages = []

        formula_markers = [
            "=",
            "loss",
            "formula",
            "equation",
            "attention",
            "softmax",
            "gradient",
            "optimization",
            "objective",
            "iou",
            "ciou",
            "loss function",
            "self-supervised",
            "contrastive",
            "infonce",
            "momentum contrast",
            "adversarial",
            "fgsm",
            "knowledge distillation",
            "learning rate",
            "convolution",
            "bounding box",
            "activation",
            "probability",
        ]

        pdf_files = sorted(
            DOCUMENTS_DIR.glob("*.pdf")
        )

        for pdf_path in pdf_files:
            try:
                document = pymupdf.open(pdf_path)

                try:
                    for page_number, page in enumerate(
                        document,
                        start=1,
                    ):
                        text = page.get_text("text").strip()

                        if not text:
                            continue

                        normalized_text = text.lower()

                        marker_count = sum(
                            marker in normalized_text
                            for marker in formula_markers
                        )

                        # Mathematical PDFs often contain formulas whose
                        # extracted text is corrupted. Therefore "=" and
                        # mathematical section terminology are sufficient
                        # evidence to keep a page for image inspection.
                        if marker_count >= 1:
                            candidate_pages.append(
                                {
                                    "source": pdf_path.name,
                                    "page_number": page_number,
                                    "text": text,
                                }
                            )

                finally:
                    document.close()

            except Exception as exc:
                print(
                    f"Unable to scan PDF {pdf_path.name}: {exc}"
                )

        print(
            f"Formula candidate-page detection found "
            f"{len(candidate_pages)} pages."
        )

        return candidate_pages

    def _build_page_context(
        self,
        page_records: list[dict],
    ) -> str:
        """Build text context for a page batch."""

        context_parts = []

        for index, page in enumerate(
            page_records,
            start=1,
        ):
            text = page["text"]

            if len(text) > 7000:
                text = text[:7000]

            context_parts.append(
                f"[Page {index}]\n"
                f"Source: {page['source']}\n"
                f"Page number: {page['page_number']}\n"
                f"Extracted text:\n{text}"
            )

        return "\n\n".join(context_parts)

    def _build_formula_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        """Build a document-wide formula extraction prompt."""

        return f"""
You are a mathematical formula extraction assistant for an academic
research document.

USER REQUEST:

{question}

You are processing ONE BATCH of original PDF pages.

Use ONLY the supplied page text and ORIGINAL PDF PAGE IMAGES.

The ORIGINAL PDF PAGE IMAGE is authoritative whenever extracted PDF text
contains corrupted mathematical glyphs.

IMPORTANT:

Extract EVERY mathematical formula or equation that is actually present
on the supplied pages.

Do NOT invent equations from general knowledge.

Do NOT replace a source formula with a standard textbook version.

Do NOT omit equations merely because they occur inside prose, tables,
methodology descriptions, architecture descriptions, loss functions,
optimization sections, self-supervised learning sections, adversarial
training sections, or other explanatory sections.

For every formula found, output exactly this structure:

### Formula

[valid LaTeX formula]

### Explanation

Explain every variable, symbol, and term that the supplied document
supports.

### Source

[Source: filename, Page: X]

Rules for mathematical notation:

- Preserve subscripts.
- Preserve superscripts.
- Preserve fractions.
- Preserve Greek symbols.
- Preserve matrices.
- Preserve summations.
- Preserve square roots.
- Preserve exponents.
- Preserve vector notation.
- Preserve tensor notation.
- Preserve hats.
- Preserve bars.
- Preserve primes.
- Preserve indices.
- Preserve mathematical operators.
- Preserve parentheses and brackets.

If extracted text and the image disagree, use the image.

If a formula is not readable with sufficient confidence from the supplied
page image, state that it could not be reconstructed reliably instead of
guessing.

Only report formulas that occur on the supplied pages.

Do not summarize unrelated prose.

PAGE CONTEXT:

{context}

Return the formulas found in this batch.
"""

    def _build_normal_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        """Build the standard RAG prompt."""

        return f"""
You are a research assistant answering questions from uploaded documents.

Use ONLY the provided document context and original PDF page images.

Rules:

1. Do not use outside knowledge.

2. Do not invent facts that are not supported by the documents.

3. If the documents do not contain enough information, say:

   "I don't have enough information in the provided documents to answer this question."

4. Answer the user's question clearly and directly.

5. Preserve the terminology and meaning used in the source document.

6. Every factual claim must include a citation using:

   [Source: filename, Page: X]

7. Use only source names and page numbers supplied in the context.

8. If mathematical notation is present, use the original PDF page image
   as the authoritative representation and render mathematics in valid
   LaTeX.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    def _build_fallback_answer(
        self,
        question: str,
        retrieved_chunks: list[dict],
    ) -> str:
        """Return grounded evidence when Gemini is unavailable."""

        if not retrieved_chunks:
            return (
                "I don't have enough information in the provided "
                "documents to answer this question."
            )

        lines = [
            "### Document-grounded evidence",
            "",
            "The AI generation service is temporarily unavailable. "
            "The following retrieved evidence is available from the "
            "uploaded documents:",
            "",
        ]

        for index, result in enumerate(
            retrieved_chunks,
            start=1,
        ):
            metadata = result["metadata"]

            source = metadata.get(
                "source",
                "Unknown source",
            )

            page_number = metadata.get(
                "page_number",
                "Unknown",
            )

            text = metadata.get(
                "text",
                "",
            ).strip()

            if not text:
                continue

            if len(text) > 1800:
                text = text[:1800].rstrip() + "..."

            lines.extend(
                [
                    f"#### Evidence {index}",
                    "",
                    text,
                    "",
                    f"[Source: {source}, Page: {page_number}]",
                    "",
                ]
            )

        return "\n".join(lines)

    def _extract_formulas_document_wide(
        self,
        question: str,
    ) -> str:
        """Extract formulas from the complete uploaded document corpus."""

        all_pages = self._get_formula_candidate_pages()

        if not all_pages:
            return (
                "I don't have any readable PDF pages available "
                "for formula extraction."
            )

        batches = [
            all_pages[index:index + FORMULA_PAGE_BATCH_SIZE]
            for index in range(
                0,
                len(all_pages),
                FORMULA_PAGE_BATCH_SIZE,
            )
        ]

        print(
            f"Formula extraction will process "
            f"{len(all_pages)} pages in "
            f"{len(batches)} batches."
        )

        batch_answers = []

        for batch_number, batch in enumerate(
            batches,
            start=1,
        ):
            print(
                f"Processing formula page batch "
                f"{batch_number}/{len(batches)}..."
            )

            context = self._build_page_context(
                batch
            )

            prompt = self._build_formula_prompt(
                question=question,
                context=context,
            )

            page_images = self._load_page_images(
                batch
            )

            batch_answer = None

            for model_name in MODEL_NAMES:
                if model_name in self.exhausted_models:
                    print(
                        f"Skipping exhausted Gemini model "
                        f"{model_name} for formula batch "
                        f"{batch_number}."
                    )
                    continue

                try:
                    print(
                        f"Trying Gemini model {model_name} "
                        f"for formula batch {batch_number}."
                    )

                    batch_answer = self._generate_with_model(
                        model_name=model_name,
                        prompt=prompt,
                        page_images=page_images,
                    )

                    print(
                        f"Formula batch {batch_number} "
                        f"succeeded with {model_name}."
                    )

                    break

                except RuntimeError as exc:
                    print(
                        f"Model {model_name} failed for "
                        f"formula batch {batch_number}: {exc}"
                    )

            if batch_answer:
                batch_answers.append(
                    batch_answer
                )
            else:
                batch_answers.append(
                    "No reliable Gemini extraction was available "
                    f"for pages "
                    f"{batch[0]['page_number']}–"
                    f"{batch[-1]['page_number']} of "
                    f"{batch[0]['source']}."
                )

        return "\n\n---\n\n".join(
            batch_answers
        )

    def generate(
        self,
        question: str,
        retrieved_chunks: list[dict],
    ) -> str:
        """Generate a grounded answer."""

        if not question.strip():
            raise ValueError(
                "Question must not be empty."
            )

        if self._is_formula_question(question):
            return self._extract_formulas_document_wide(
                question=question
            )

        if not retrieved_chunks:
            return (
                "I don't have enough information in the provided "
                "documents to answer this question."
            )

        context = self._build_page_context(
            [
                {
                    "source": result["metadata"]["source"],
                    "page_number": result["metadata"]["page_number"],
                    "text": result["metadata"]["text"],
                }
                for result in retrieved_chunks
            ]
        )

        prompt = self._build_normal_prompt(
            question=question,
            context=context,
        )

        page_images = self._load_page_images(
            [
                {
                    "source": result["metadata"]["source"],
                    "page_number": result["metadata"]["page_number"],
                }
                for result in retrieved_chunks
            ]
        )

        for model_name in MODEL_NAMES:
            if model_name in self.exhausted_models:
                print(
                    f"Skipping exhausted Gemini model: "
                    f"{model_name}"
                )
                continue

            try:
                print(
                    f"Trying Gemini model: {model_name}"
                )

                answer = self._generate_with_model(
                    model_name=model_name,
                    prompt=prompt,
                    page_images=page_images,
                )

                print(
                    f"Gemini model succeeded: {model_name}"
                )

                return answer

            except RuntimeError as exc:
                print(
                    f"Model {model_name} unavailable: {exc}"
                )

        return self._build_fallback_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )