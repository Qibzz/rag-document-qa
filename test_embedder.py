from ingest.document_loader import build_chunks
from embed.embedder import Embedder

chunks = build_chunks("sample_docs/test.txt")

embedder = Embedder()  # load model dulu (agak lama pas pertama kali, download model)
result = embedder.embed_chunks(chunks, output_dir="storage")

print(f"Berhasil! {result['embedding_count']} embeddings dibuat")
print(f"Dimensi tiap embedding: {result['embedding_dim']}")
print(f"Disimpan di: {result['vector_path']}")