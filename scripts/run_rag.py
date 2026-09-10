from app.retrieval import hybrid_retriever
from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.generation.rag_generator import generate_answer


store = ChromaVectorStore()

hybrid_retriever = HybridRetriever(
    chroma_store = store
)

reranker = Reranker()

query = (
    "What treatment is recommended "
    "for migraine prevention?"
)

candidates = hybrid_retriever.search(
    query=query,
    top_k=5,
)

retrieved_chunks = reranker.rerank(
    query=query,
    candidates=candidates,
    top_k=3,
)

answer = generate_answer(
    query=query,
    retrieved_chunks = retrieved_chunks,

)

print("\nQUESTION:")
print(query)

print("\nANSWER:")
print(answer)