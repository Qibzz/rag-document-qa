from retrieve.retriever import Retriever

retriever = Retriever(embeddings_path="storage/embeddings.npy")

hasil = retriever.retrieve("apakah chunking berjalan dengan baik?", top_k=1)
for r in hasil:
    print(f"Score: {r['score']:.4f}")
    print(f"Source: {r['source']}")
    print(f"Content: {r['content']}")