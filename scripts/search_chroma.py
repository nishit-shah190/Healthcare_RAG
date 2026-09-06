from app.retrieval.chroma_store import ChromaVectorStore

store = ChromaVectorStore()

results = store.search(
    query = "How should diabetic patients be treated?",
    top_k =3,
    where = { "document_id" : "diabetes_guideline"}
)

for i in range(len(results["documents"][0])):
    print("=" * 60)
    print("Documents:", results["metadatas"][0][i]["document_id"])
    print("Distance:", results["distances"][0][i])
    print(results["documents"][0][i])


