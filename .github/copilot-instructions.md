# Project: Simple RAG-based Document Q&A Assistant

## Context
Portfolio project untuk internship AI role. Tujuannya: sistem simple tapi 
scalable yang bisa jawab pertanyaan berdasarkan isi dokumen yang di-upload, 
menggunakan pendekatan RAG (Retrieval-Augmented Generation) ringan — bukan 
full vector database, cukup embedding sederhana + cosine similarity.

## Architecture (modular, jangan monolithic)
project/
├── ingest/
│   └── document_loader.py     # load & chunk dokumen (PDF/txt)
├── embed/
│   └── embedder.py             # generate embeddings per chunk
├── retrieve/
│   └── retriever.py            # cari chunk paling relevan (cosine similarity)
├── generate/
│   └── qa_engine.py            # panggil LLM API, gabungin context + jawab
├── app/
│   └── main.py                 # Flask app, endpoint upload & ask
├── config/
│   └── settings.py             # API keys via environment variables
├── requirements.txt
└── README.md

## Requirements per tahap

### 1. Ingest
- Load dokumen PDF/txt, chunking dengan overlap (misal 500 char per chunk, 
  50 char overlap)
- Simpan chunk + metadata (source, chunk_id) ke local storage (JSON/SQLite, 
  gak perlu vector DB berbayar)

### 2. Embed
- Generate embedding tiap chunk pakai sentence-transformers (model kecil, 
  free, jalan lokal — misal all-MiniLM-L6-v2)
- Simpan embedding ke file (.npy atau SQLite)

### 3. Retrieve
- Terima query dari user, generate embedding query-nya
- Hitung cosine similarity ke semua chunk embeddings
- Return top-k chunk paling relevan

### 4. Generate
- Gabungin retrieved chunks jadi context
- Kirim ke LLM API (Claude API atau OpenAI, whichever available) dengan 
  prompt: "jawab pertanyaan ini HANYA berdasarkan context berikut..."
- Return jawaban + source chunk yang dipakai (biar transparent, gak halusinasi)

### 5. App (Flask, bukan Streamlit/Gradio)
- Endpoint POST /upload — terima dokumen
- Endpoint POST /ask — terima pertanyaan, return jawaban + sumber
- UI sederhana: form upload + chat box

## Coding standards
- Docstring tiap function
- Environment variables via python-dotenv (API keys jangan hardcode)
- Error handling: dokumen kosong, API timeout, query kosong
- Logging pakai modul `logging`

## Deliverable
- Demo lokal yang bisa di-screen-record: upload dokumen -> tanya -> 
  dapet jawaban + sumbernya
- README jelasin arsitektur RAG-nya (diagram simple pake Mermaid), 
  cara run, contoh Q&A
- Bonus: deploy ke Render biar ada live link

Mulai dari ingest module dulu — kasih kode lengkap document_loader.py 
dengan chunking logic dan error handling.