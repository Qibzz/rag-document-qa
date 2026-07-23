from ingest.document_loader import build_chunks, save_chunks_to_json

chunks = build_chunks("sample_docs/test.txt")
save_chunks_to_json(chunks, "storage/chunks.json")
print(f"Berhasil! {len(chunks)} chunks dibuat")