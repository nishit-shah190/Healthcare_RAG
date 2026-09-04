from app.retrieval.vector_store import LocalVectorStore


store = LocalVectorStore()


query = (
  "What should be considered for diabetes patients with kidney problems?"
)


results = store.search(
    query=query,
    top_k=3,
)



for result in results:

    print("=" * 60)

    print(
        "Score:",
        round(
            result["score"],
            3,
        ),
    )

    print(
        "Document:",
        result["chunk"].document_id,
    )

    print(
        result["chunk"].content
    )