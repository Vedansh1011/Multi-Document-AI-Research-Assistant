import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:8000";

function normalizeMathDelimiters(text) {
  return text
    .replace(/\\\[/g, "$$")
    .replace(/\\\]/g, "$$")
    .replace(/\\\(/g, "$")
    .replace(/\\\)/g, "$");
}

function App() {
  const [file, setFile] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [question, setQuestion] = useState("");
  const [uploadStatus, setUploadStatus] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [deletingDocument, setDeletingDocument] = useState("");
  const [expandedSource, setExpandedSource] = useState(null);

  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem("theme") === "dark";
  });

  useEffect(() => {
    document.documentElement.setAttribute(
      "data-theme",
      darkMode ? "dark" : "light"
    );

    localStorage.setItem(
      "theme",
      darkMode ? "dark" : "light"
    );
  }, [darkMode]);

  const loadDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Unable to load documents.");
      }

      setDocuments(data.documents || []);
    } catch (error) {
      setUploadStatus(error.message);
    }
  };

  useEffect(() => {
    loadDocuments();
  }, []);

  const uploadDocument = async () => {
    if (!file) return;

    setLoading(true);
    setUploadStatus("");
    setAnswer("");
    setSources([]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Upload failed.");
      }

      setUploadStatus(`${data.filename} added successfully.`);
      setFile(null);

      await loadDocuments();
    } catch (error) {
      setUploadStatus(error.message);
    } finally {
      setLoading(false);
    }
  };

  const deleteDocument = async (filename) => {
    const confirmed = window.confirm(
      `Remove "${filename}" from the research assistant?`
    );

    if (!confirmed) return;

    setDeletingDocument(filename);
    setUploadStatus("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/documents/${encodeURIComponent(filename)}`,
        {
          method: "DELETE",
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Unable to remove document.");
      }

      setDocuments((currentDocuments) =>
        currentDocuments.filter(
          (document) => document.filename !== filename
        )
      );

      setUploadStatus(`${filename} removed successfully.`);
      setAnswer("");
      setSources([]);
    } catch (error) {
      setUploadStatus(error.message);
    } finally {
      setDeletingDocument("");
    }
  };

  const askQuestion = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setAnswer("");
    setSources([]);

    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          top_k: 5,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Unable to generate an answer.");
      }

      setAnswer(data.answer);
      setSources(data.sources || []);
    } catch (error) {
      setAnswer(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app">
      <section className="container">

        <div className="theme-toggle-wrapper">
          <button
            className="theme-toggle"
            onClick={() => setDarkMode((current) => !current)}
            aria-label="Toggle theme"
          >
            {darkMode ? "☀️" : "🌙"}
          </button>
        </div>

        <header className="hero">
          <div className="hero-brand">
            <div className="brand-icon">📖</div>

            <div>
              <p className="eyebrow">AI RESEARCH ASSISTANT</p>

              <h1>AI Research Assistant</h1>

              <p>
                Upload research documents, retrieve relevant evidence,
                and ask grounded questions with source citations.
              </p>
            </div>
          </div>
        </header>

        <section className="card">
          <h2>1. Upload documents</h2>

          <div className="upload-row">
            <input
              type="file"
              accept=".pdf"
              onChange={(event) =>
                setFile(event.target.files[0] || null)
              }
            />

            <button
              onClick={uploadDocument}
              disabled={!file || loading}
            >
              {loading ? "Processing..." : "Upload PDF"}
            </button>
          </div>

          {uploadStatus && (
            <p className="status">{uploadStatus}</p>
          )}
        </section>

        <section className="card">
          <div className="section-heading">
            <div>
              <h2>2. Uploaded documents</h2>

              <p className="section-description">
                Documents currently available to the research assistant.
              </p>
            </div>

            <span className="document-count">
              {documents.length}{" "}
              {documents.length === 1
                ? "document"
                : "documents"}
            </span>
          </div>

          {documents.length === 0 ? (
            <p className="empty-state">
              No documents indexed yet. Upload a PDF to get started.
            </p>
          ) : (
            <div className="documents">
              {documents.map((document) => (
                <article
                  key={document.filename}
                  className="document-card"
                >
                  <div className="document-info">
                    <span className="document-icon">
                      PDF
                    </span>

                    <div>
                      <strong>{document.filename}</strong>
                      <p>Added</p>
                    </div>
                  </div>

                  <button
                    className="delete-button"
                    onClick={() =>
                      deleteDocument(document.filename)
                    }
                    disabled={
                      deletingDocument === document.filename
                    }
                    aria-label={`Remove ${document.filename}`}
                    title={`Remove ${document.filename}`}
                  >
                    {deletingDocument === document.filename
                      ? "..."
                      : "×"}
                  </button>
                </article>
              ))}
            </div>
          )}
        </section>

        <section className="card">
          <h2>3. Ask a question</h2>

          <textarea
            value={question}
            onChange={(event) =>
              setQuestion(event.target.value)
            }
            placeholder="Ask something about the uploaded documents..."
            rows="4"
          />

          <button
            className="primary-button"
            onClick={askQuestion}
            disabled={!question.trim() || loading}
          >
            {loading
              ? "Generating..."
              : "Ask Research Assistant"}
          </button>
        </section>

        {answer && (
          <section className="card answer-card">
            <h2>Answer</h2>

            <div className="answer">
              <ReactMarkdown
                remarkPlugins={[remarkMath]}
                rehypePlugins={[rehypeKatex]}
              >
                {normalizeMathDelimiters(answer)}
              </ReactMarkdown>
            </div>
          </section>
        )}

        {sources.length > 0 && (
          <section className="card">
            <h2>Sources</h2>

            <div className="sources">
              {sources.map((source) => (
                <article
                  key={source.chunk_id}
                  className="source"
                >
                  <div className="source-header">
                    <strong>{source.source}</strong>

                    <span>
                      Page {source.page_number}
                    </span>
                  </div>

                  <p className="source-text">
                    {source.text}
                  </p>

                  <button
                    className="source-preview-button"
                    onClick={() =>
                      setExpandedSource(
                        expandedSource === source.chunk_id
                          ? null
                          : source.chunk_id
                      )
                    }
                  >
                    {expandedSource === source.chunk_id
                      ? "Hide Original Page"
                      : "View Original Page"}
                  </button>

                  {expandedSource === source.chunk_id && (
                    <div className="source-page-preview">
                      <img
                        src={`${API_BASE_URL}/documents/${encodeURIComponent(
                          source.source
                        )}/pages/${source.page_number}`}
                        alt={`Page ${source.page_number} of ${source.source}`}
                      />
                    </div>
                  )}
                </article>
              ))}
            </div>
          </section>
        )}

      </section>
    </main>
  );
}

export default App;