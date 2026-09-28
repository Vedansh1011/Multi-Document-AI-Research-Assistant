import { useState } from "react";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [uploadStatus, setUploadStatus] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);

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

      setUploadStatus(
        `${data.filename} indexed successfully — ${data.chunks_indexed} chunks.`
      );
    } catch (error) {
      setUploadStatus(error.message);
    } finally {
      setLoading(false);
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
        <header className="hero">
          <p className="eyebrow">AI RESEARCH ASSISTANT</p>
          <h1>Multi-Document RAG</h1>
          <p>
            Upload research documents, retrieve relevant evidence, and ask
            grounded questions with source citations.
          </p>
        </header>

        <section className="card">
          <h2>1. Upload a document</h2>

          <div className="upload-row">
            <input
              type="file"
              accept=".pdf"
              onChange={(event) => setFile(event.target.files[0])}
            />

            <button onClick={uploadDocument} disabled={!file || loading}>
              {loading ? "Processing..." : "Upload PDF"}
            </button>
          </div>

          {uploadStatus && <p className="status">{uploadStatus}</p>}
        </section>

        <section className="card">
          <h2>2. Ask a question</h2>

          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask something about the uploaded documents..."
            rows="4"
          />

          <button
            className="primary-button"
            onClick={askQuestion}
            disabled={!question.trim() || loading}
          >
            {loading ? "Generating..." : "Ask Research Assistant"}
          </button>
        </section>

        {answer && (
          <section className="card answer-card">
            <h2>Answer</h2>
            <p className="answer">{answer}</p>
          </section>
        )}

        {sources.length > 0 && (
          <section className="card">
            <h2>Sources</h2>

            <div className="sources">
              {sources.map((source) => (
                <article key={source.chunk_id} className="source">
                  <div className="source-header">
                    <strong>{source.source}</strong>
                    <span>Page {source.page_number}</span>
                  </div>

                  <p>
                    {source.text.length > 350
                      ? `${source.text.slice(0, 350)}...`
                      : source.text}
                  </p>
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