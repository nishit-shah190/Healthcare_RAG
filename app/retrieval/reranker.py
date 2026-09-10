from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(
        self,
        model_name: str = (
            "cross-encoder/"
            "ms-marco-MiniLM-L-6-v2"
        ),
    ):
        self.model = CrossEncoder(
            model_name
        )
    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 3,
    ):
        pairs = [
            [query, candidate["document"]]
            for candidate in candidates
        ]

        scores = self.model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        candidates.sort(
            key=lambda candidate: candidate["rerank_score"],
            reverse=True,
        )

        return candidates[:top_k]
