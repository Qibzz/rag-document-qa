"""Flask application for the lightweight RAG document Q&A demo.

Routes:
- GET /: simple HTML page with upload and chat form.
- POST /upload: upload a document file, chunk it, and generate embeddings.
- POST /ask: accept a question, retrieve top-k chunks, and generate an answer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template_string, request

from embed.embedder import Embedder
from generate.qa_engine import QAEngine
from ingest.document_loader import build_chunks, save_chunks_to_json
from retrieve.retriever import Retriever

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
EMBEDDINGS_PATH = STORAGE_DIR / "embeddings.npy"
METADATA_PATH = STORAGE_DIR / "chunk_metadata.json"
CHUNKS_JSON_PATH = STORAGE_DIR / "chunks.json"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

UPLOAD_HTML = """
<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Document Q&A</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', Arial, sans-serif;
      margin: 0;
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      min-height: 100vh;
      padding: 40px 20px;
      color: #e2e8f0;
    }
    .container { max-width: 720px; margin: auto; }
    h1 {
      text-align: center;
      font-size: 2rem;
      margin-bottom: 8px;
      background: linear-gradient(90deg, #38bdf8, #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .subtitle {
      text-align: center;
      color: #94a3b8;
      margin-bottom: 32px;
      font-size: 0.95rem;
    }
    .card {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 20px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .card h2 {
      font-size: 1.1rem;
      margin-top: 0;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .step-num {
      background: #38bdf8;
      color: #0f172a;
      width: 24px;
      height: 24px;
      border-radius: 50%;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 0.85rem;
      font-weight: bold;
    }
    input[type="file"] {
      color: #e2e8f0;
      margin-bottom: 12px;
      display: block;
    }
    textarea {
      width: 100%;
      padding: 12px 14px;
      border-radius: 10px;
      border: 1px solid #334155;
      background: #0f172a;
      color: #e2e8f0;
      font-size: 0.95rem;
      resize: vertical;
      margin-bottom: 12px;
    }
    textarea:focus, input:focus { outline: 2px solid #38bdf8; }
    button {
      background: linear-gradient(90deg, #38bdf8, #6366f1);
      color: white;
      border: none;
      padding: 10px 20px;
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.2s;
    }
    button:hover { opacity: 0.85; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .result-box {
      margin-top: 16px;
      padding: 14px;
      border-radius: 10px;
      background: #0f172a;
      border-left: 3px solid #38bdf8;
      font-size: 0.9rem;
      white-space: pre-wrap;
      word-break: break-word;
      display: none;
    }
    .result-box.show { display: block; }
    .result-box.error { border-left-color: #f87171; }
    .answer-text {
      font-size: 1rem;
      line-height: 1.6;
      margin-bottom: 12px;
    }
    .sources {
      font-size: 0.8rem;
      color: #94a3b8;
      border-top: 1px solid #334155;
      padding-top: 10px;
      margin-top: 10px;
    }
    .loading { color: #38bdf8; font-style: italic; }
  </style>
</head>
<body>
  <div class="container">
    <h1>RAG Document Q&A</h1>
    <p class="subtitle">Upload dokumen, tanya apa aja, jawaban langsung dari isi dokumennya</p>

    <div class="card">
      <h2><span class="step-num">1</span> Upload Dokumen</h2>
      <form id="uploadForm" enctype="multipart/form-data">
        <input type="file" name="file" accept=".txt,.pdf" required>
        <button type="submit" id="uploadBtn">Upload & Embed</button>
      </form>
      <div id="uploadResult" class="result-box"></div>
    </div>

    <div class="card">
      <h2><span class="step-num">2</span> Tanya Jawab</h2>
      <form id="askForm">
        <textarea name="question" rows="3" placeholder="Tulis pertanyaan kamu di sini..." required></textarea>
        <button type="submit" id="askBtn">Tanya</button>
      </form>
      <div id="answer" class="result-box"></div>
    </div>
  </div>

  <script>
    const uploadForm = document.getElementById('uploadForm');
    const askForm = document.getElementById('askForm');
    const uploadResult = document.getElementById('uploadResult');
    const answerBox = document.getElementById('answer');

    uploadForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const btn = document.getElementById('uploadBtn');
      btn.disabled = true;
      uploadResult.className = 'result-box show';
      uploadResult.innerHTML = '<span class="loading">Uploading & generating embeddings...</span>';

      const formData = new FormData(uploadForm);
      try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const result = await response.json();
        if (result.error) {
          uploadResult.className = 'result-box show error';
          uploadResult.textContent = 'Error: ' + result.error;
        } else {
          uploadResult.className = 'result-box show';
          uploadResult.textContent = `✓ ${result.message}\\n${result.chunks_created} chunks dibuat dari ${result.source}`;
        }
      } catch (err) {
        uploadResult.className = 'result-box show error';
        uploadResult.textContent = 'Error: ' + err.message;
      }
      btn.disabled = false;
    });

    askForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const btn = document.getElementById('askBtn');
      btn.disabled = true;
      answerBox.className = 'result-box show';
      answerBox.innerHTML = '<span class="loading">Mikir dulu...</span>';

      const formData = new FormData(askForm);
      const payload = { question: formData.get('question') };

      try {
        const response = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const result = await response.json();

        if (result.error) {
          answerBox.className = 'result-box show error';
          answerBox.textContent = 'Error: ' + result.error;
        } else {
          const sourcesHtml = result.sources.map(s =>
            `${s.chunk_id} (score: ${s.score.toFixed(3)})`
          ).join(', ');
          answerBox.className = 'result-box show';
          answerBox.innerHTML = `<div class="answer-text">${result.answer}</div><div class="sources">Sumber: ${sourcesHtml}</div>`;
        }
      } catch (err) {
        answerBox.className = 'result-box show error';
        answerBox.textContent = 'Error: ' + err.message;
      }
      btn.disabled = false;
    });
  </script>
</body>
</html>
"""


def _ensure_storage() -> None:
    """Create local storage folders required by the demo pipeline."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _save_uploaded_file(file_storage) -> Path:
    """Persist the uploaded document to disk and return the saved path."""
    _ensure_storage()
    filename = Path(file_storage.filename or "upload.txt").name
    target = UPLOAD_DIR / filename
    file_storage.save(target)
    return target


@app.get("/")
def index() -> str:
    """Render a minimal HTML interface for upload and Q&A."""
    return render_template_string(UPLOAD_HTML)


@app.post("/upload")
def upload_document():
    """Upload a document, chunk it, and generate embeddings for retrieval."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file_storage = request.files["file"]
    if file_storage.filename == "":
        return jsonify({"error": "No selected file."}), 400

    try:
        source_path = _save_uploaded_file(file_storage)
        chunks = build_chunks(source_path)
        save_chunks_to_json(chunks, CHUNKS_JSON_PATH)

        embedder = Embedder()
        result = embedder.embed_chunks(chunks, STORAGE_DIR)

        return jsonify(
            {
                "status": "success",
                "message": "Dokumen berhasil diupload, di-chunk, dan di-embed.",
                "source": str(source_path),
                "chunks_created": len(chunks),
                "embedding_path": result["vector_path"],
                "metadata_path": result["metadata_path"],
            }
        )
    except Exception as exc:
        logger.exception("Upload failed")
        return jsonify({"error": str(exc)}), 500


@app.post("/ask")
def ask_question():
    """Answer a user question using retrieval and the Gemini QA engine."""
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "")

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Question is required."}), 400

    try:
        if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
            return jsonify({"error": "No uploaded document embedding available yet. Please upload a document first."}), 400

        retriever = Retriever(EMBEDDINGS_PATH, METADATA_PATH)
        ranked_chunks = retriever.retrieve(question, top_k=3)

        qa_engine = QAEngine()
        answer_result = qa_engine.generate_answer(question, ranked_chunks)

        return jsonify(
            {
                "question": question,
                "answer": answer_result["answer"],
                "sources": answer_result["sources"],
                "context": answer_result["context"],
            }
        )
    except Exception as exc:
        logger.exception("Ask failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
