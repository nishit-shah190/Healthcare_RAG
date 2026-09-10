from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.hybrid_retriever import HybridRetriever

store = ChromaVectorStore()

retriever = HybridRetriever(
    chroma_store=store
)


queries = [
    "What lifestyle changes help hypertension?",
    "What does the guideline say about metformin?",
    "What should be considered for diabetes with kidney problems?",
]

for query in queries:
    print("\n" + "=" * 80)
    print("QUERY:", query)


    results = retriever.search(
        query = query,
        top_k =5,
    )

    for result in results:
        print("-" * 60)
        print(
            "Document:",
            result["metadata"]["document_id"]

        )
        print(
            "RRF:",
            round(
                result["rrf_score"], 4
            )
        )

        print(result["document"])

