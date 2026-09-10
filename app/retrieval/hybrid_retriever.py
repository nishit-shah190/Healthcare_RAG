from rank_bm25 import BM25Okapi

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

        self.bm25 = BM25Okapi(
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
        return results


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

        for i in range(len(results["documents"][0])):
            dense_results.append({
                "id":results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata":results["metadatas"][0][i],
                "distance":results["distances"][0][i]
            }
                

            )
        
        return dense_results

    def reciprocal_rank_fusion(
        self, 
        dense_results,
        bm25_results,
        k: int =60,
    ):
        fused_scores = {}
        candidate_data = {}

        for rank, result in enumerate( dense_results, start=1,):
            chunk_id = result["id"]
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id,0) + 1 / (k+rank)
            )
            candidate_data[chunk_id] = result

        for rank, result in enumerate(bm25_results, start=1,):
            chunk_id = result["id"]
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0) + 1 / (k+rank)
            )
            candidate_data[chunk_id]=result

        ranked_ids = sorted(
            fused_scores,
            key =fused_scores.get,
            reverse=True,
        )

        results= []

        for chunk_id in ranked_ids:
            result = candidate_data[chunk_id].copy()
            result["rrf_score"] = (
                fused_scores[chunk_id]
            )

            results.append(result)

        return results
    

    def search(
        self,
        query: str,
        top_k: int =5,
    ):
        dense_results = self.dense_search(
            query = query,
            top_k = top_k,
        )

        bm25_results = self.bm25_search(
            query=query,
            top_k=top_k,
        )

        fused_results = (
            self.reciprocal_rank_fusion(
                dense_results,
                bm25_results,
            )
        )

        return fused_results[:top_k]



