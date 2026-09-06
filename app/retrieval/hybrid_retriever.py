from rank_bm25 import BM250kapi

from app.retrieval.chroma_store import ChromaVectorStore

class HybridRetriever:
    def __init__(
        self,
        chroma_store: ChromaVectorStore,
    ):
        self.chroma_store = chroma_store

        records = self.chroma_store.collection.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

        self.ids = records["ids"]
        self.documents = records["documents"]
        self.metadatas = records["metadatas"]

        tokenized_document = [
            document.lower().split()
            for document in self.documents
        ] 

        self.bm25 = BM250kapi(
            tokenized_document
        )

    def bm25_search(
        self,
        query: str,
        top_k: int =5,
    ):
        tokenized_query = (
            query.lower().split()
        )

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key = lambda i: scores[i],
            reverse = True,
        )

        results = []

        for index in ranked_indices[:top_k]:
            results.append(
                {
                    "id": self.ids[index],
                    "document": self.documents[index],
                    "metadata": self.metadatas[index],
                    "score": float(scores[index])
                }
            )


    def dense_search(
        self,
        query: str,
        top_k: int =5,
    ):
        results = self.chroma_store.search(
            query=query,
            top_k=top_k,
        )

        dense_results = []

        for i in range(len(results["documets"][0])):
            dense_results.append({
                "id":results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata":results["metadatas"][0][i],
                "distance":results["distances"][0][i]
            }
                

            )

        return dense_results

