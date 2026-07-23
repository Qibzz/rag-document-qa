from dotenv import load_dotenv
load_dotenv()

from retrieve.retriever import Retriever
from generate.qa_engine import QAEngine

retriever = Retriever(embeddings_path="storage/embeddings.npy")
hasil_retrieval = retriever.retrieve("apakah chunking berjalan dengan baik?", top_k=1)

qa = QAEngine()
result = qa.generate_answer(
    question="apakah chunking berjalan dengan baik?",
    retrieved_chunks=hasil_retrieval
)

print("JAWABAN:", result["answer"])
print("\nSOURCES:", result["sources"])