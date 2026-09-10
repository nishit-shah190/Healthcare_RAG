from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker


store = ChromaVectorStore()

retriever = HybridRetriever(
    chroma_store=store
)

reranker = Reranker()


query = (
    "What should be considered "
    "for diabetes patients with "
    "kidney problems?"
)


candidates = retriever.search(
    query=query,
    top_k=5,
)


results = reranker.rerank(
    query=query,
    candidates=candidates,
    top_k=3,
)


for result in results:

    print("=" * 60)

    print(
        "Document:",
        result["metadata"]["document_id"]
    )

    print(
        "RRF score:",
        round(
            result["rrf_score"],
            4
        )
    )

    print(
        "Rerank score:",
        round(
            result["rerank_score"],
            4
        )
    )

    print(
        result["document"]
    )