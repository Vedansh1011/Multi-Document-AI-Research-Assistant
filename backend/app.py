from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import re

from backend.embedding import EmbeddingModel
from backend.generator import AnswerGenerator
from backend.pdf_processor import extract_text_from_pdf
from backend.retrieval import Retriever
from backend.text_processor import chunk_pages
from backend.vector_store import VectorStore


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


embedding_model = EmbeddingModel()
generator = AnswerGenerator()

PROCESSED_DIR = Path("data/processed")
INDEX_PATH = PROCESSED_DIR / "vector_store.faiss"
METADATA_PATH = PROCESSED_DIR / "vector_metadata.pkl"

vector_store: VectorStore | None = None
retriever: Retriever | None = None

@app.get("/")
def home():
    return {
        "message": "Multi-Document AI Research Assistant API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "indexed": vector_store is not None
        and vector_store.index.ntotal > 0,
        "total_chunks": (
            vector_store.index.ntotal
            if vector_store is not None
            else 0
        ),
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

    documents_dir = Path("data/documents")
    documents_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_filename = Path(file.filename).name
    file_path = documents_dir / safe_filename

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

def clean_source_text(text: str) -> str:
    """Clean obvious PDF extraction artifacts for source display."""
    text = text.replace("\ufffd", "")
    text = re.sub(r"[\u0b00-\u0b7f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

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

    retrieved_chunks = retriever.retrieve(
        request.question,
        top_k=request.top_k,
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
                "text": clean_source_text(result["metadata"]["text"]),
            }
            for result in retrieved_chunks
        ],
    }