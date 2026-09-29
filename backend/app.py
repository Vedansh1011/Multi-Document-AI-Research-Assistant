from contextlib import asynccontextmanager
from pathlib import Path
import re
import unicodedata

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pymupdf

from backend.embedding import EmbeddingModel
from backend.generator import AnswerGenerator
from backend.pdf_processor import extract_text_from_pdf
from backend.retrieval import Retriever
from backend.text_processor import chunk_pages
from backend.vector_store import VectorStore


PROCESSED_DIR = Path("data/processed")
DOCUMENTS_DIR = Path("data/documents")

INDEX_PATH = PROCESSED_DIR / "vector_store.faiss"
METADATA_PATH = PROCESSED_DIR / "vector_metadata.pkl"

embedding_model = EmbeddingModel()
generator = AnswerGenerator()

vector_store: VectorStore | None = None
retriever: Retriever | None = None


def load_persisted_store() -> None:
    """Load the persisted vector store if it exists."""
    global vector_store
    global retriever

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        return

    vector_store = VectorStore.load(
        INDEX_PATH,
        METADATA_PATH,
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the persisted document index when the application starts."""
    load_persisted_store()
    yield


app = FastAPI(
    title="Multi-Document AI Research Assistant",
    description="RAG-powered research assistant for multi-document question answering.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/")
def home():
    return {
        "message": "Multi-Document AI Research Assistant API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "indexed": (
            vector_store is not None
            and vector_store.index.ntotal > 0
        ),
        "total_chunks": (
            vector_store.index.ntotal
            if vector_store is not None
            else 0
        ),
    }


@app.get("/documents")
def list_documents():
    """Return all currently indexed documents."""

    if vector_store is None:
        return {
            "documents": []
        }

    documents = {}

    for metadata in vector_store.metadata:
        source = metadata.get("source")

        if not source:
            continue

        if source not in documents:
            documents[source] = {
                "filename": source,
                "chunks": 0,
            }

        documents[source]["chunks"] += 1

    return {
        "documents": list(documents.values())
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store
    global retriever

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_filename = Path(file.filename).name
    file_path = DOCUMENTS_DIR / safe_filename

    contents = await file.read()
    file_path.write_bytes(contents)

    try:
        pages = extract_text_from_pdf(file_path)

        chunks = chunk_pages(
            pages,
            chunk_size=200,
            chunk_overlap=40,
        )

        if not chunks:
            raise ValueError(
                "No searchable text was extracted from the PDF."
            )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = embedding_model.encode(texts)

        if vector_store is None:
            vector_store = VectorStore(
                dimension=embeddings.shape[1]
            )

        if embeddings.shape[1] != vector_store.dimension:
            raise ValueError(
                "Embedding dimension does not match the existing vector store."
            )

        vector_store.add(
            embeddings,
            chunks,
        )

        retriever = Retriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        vector_store.save(
            INDEX_PATH,
            METADATA_PATH,
        )

        return {
            "message": "Document indexed successfully.",
            "filename": file.filename,
            "pages_indexed": len(pages),
            "chunks_indexed": len(chunks),
            "total_chunks": vector_store.index.ntotal,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Document processing failed. Please check the PDF and try again.",
        ) from exc


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Remove a document from the index and stored document directory."""

    global vector_store
    global retriever

    safe_filename = Path(filename).name

    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF documents can be deleted.",
        )

    if vector_store is None:
        raise HTTPException(
            status_code=404,
            detail="No indexed documents found.",
        )

    removed_chunks = vector_store.delete_source(
        safe_filename
    )

    if removed_chunks == 0:
        raise HTTPException(
            status_code=404,
            detail="Document not found in the index.",
        )

    vector_store.save(
        INDEX_PATH,
        METADATA_PATH,
    )

    retriever = (
        Retriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
        )
        if vector_store.index.ntotal > 0
        else None
    )

    file_path = DOCUMENTS_DIR / safe_filename

    if file_path.exists():
        file_path.unlink()

    return {
        "message": "Document removed successfully.",
        "filename": safe_filename,
        "chunks_removed": removed_chunks,
        "remaining_chunks": vector_store.index.ntotal,
    }


def clean_source_text(text: str) -> str:
    """Clean PDF extraction artifacts and repair known mathematical formulas."""

    import unicodedata

    text = unicodedata.normalize("NFKC", text)

    # Remove Unicode replacement characters.
    text = text.replace("\ufffd", "")

    # Repair mathematical expressions corrupted by the PDF font encoding.
    formula_replacements = {
        "Attention(Q, K, V) = softmax": (
            "Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V"
        ),
        "LCIoU = 1 - IoU +": (
            "LCIoU = 1 − IoU + ρ²(b, bᵍᵗ) / c² + αv"
        ),
        "υ =": (
            "υ = (4 / π²) [arctan(wᵍᵗ / hᵍᵗ) − arctan(w / h)]²"
        ),
        "xadv = x +": (
            "xadv = x + ε · sign(∇ₓJ(θ, x, y))"
        ),
    }

    for corrupted, corrected in formula_replacements.items():
        if corrupted in text:
            start = text.find(corrupted)

            end_markers = [
                "\n",
                "Here,",
                "In this formula",
                "The variable",
            ]

            end = len(text)

            for marker in end_markers:
                marker_position = text.find(
                    marker,
                    start + len(corrupted),
                )

                if marker_position != -1:
                    end = min(end, marker_position)

            text = (
                text[:start]
                + corrected
                + text[end:]
            )

    # Remove private-use characters produced by embedded PDF fonts.
    text = re.sub(r"[\ue000-\uf8ff]", "", text)

    # Remove supplementary private-use characters.
    text = re.sub(
        r"[\U000f0000-\U000ffffd]",
        "",
        text,
    )

    # Remove control characters while preserving normal whitespace.
    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        "",
        text,
    )

    # Remove the previously observed unwanted extraction range.
    text = re.sub(r"[\u0b00-\u0b7f]", "", text)

    # Normalize repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()

@app.get("/documents/{filename}/pages/{page_number}")
def get_document_page(filename: str, page_number: int):
    """Render an original PDF page for faithful source display."""

    safe_filename = Path(filename).name

    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF documents are supported.",
        )

    if page_number <= 0:
        raise HTTPException(
            status_code=400,
            detail="Page number must be greater than 0.",
        )

    file_path = DOCUMENTS_DIR / safe_filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    try:
        document = pymupdf.open(file_path)

        try:
            if page_number > len(document):
                raise HTTPException(
                    status_code=404,
                    detail="Page not found.",
                )

            page = document[page_number - 1]

            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(1.5, 1.5),
                alpha=False,
            )

            image_bytes = pixmap.tobytes("png")

        finally:
            document.close()

        return Response(
            content=image_bytes,
            media_type="image/png",
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to render the PDF page.",
        ) from exc

@app.post("/ask")
def ask_question(request: QuestionRequest):
    if retriever is None:
        raise HTTPException(
            status_code=400,
            detail="No document has been indexed. Upload a PDF first.",
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question must not be empty.",
        )

    if request.top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="top_k must be greater than 0.",
        )

    formula_keywords = [
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

    is_formula_question = any(
        keyword in request.question.lower()
        for keyword in formula_keywords
    )

    retrieval_top_k = (
        min(50, vector_store.index.ntotal)
        if is_formula_question
        else request.top_k
    )

    retrieved_chunks = retriever.retrieve(
        request.question,
        top_k=retrieval_top_k,
    )

    try:
        answer = generator.generate(
            request.question,
            retrieved_chunks,
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    return {
        "question": request.question,
        "answer": answer,
        "sources": [
            {
                "chunk_id": result["metadata"]["chunk_id"],
                "source": result["metadata"]["source"],
                "page_number": result["metadata"]["page_number"],
                "score": round(result["score"], 4),
                "text": clean_source_text(
                    result["metadata"]["text"]
                ),
            }
            for result in retrieved_chunks
        ],
    }