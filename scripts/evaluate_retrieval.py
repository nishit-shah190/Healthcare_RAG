import json
from pathlib import Path

from app.retrieval import hybrid_retriever
from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.evaluation.retrieval_metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)


eval_path = Path("app/evaluation/retrieval_eval.json")

with open(
    eval_path,
    "r",
    encoding = "utf-8",
) as file:
    evaluation_data = json.load(file)

store = ChromaVectorStore()
hybrid_retriever = HybridRetriever(store)
reranker = Reranker()


def dense_retrieve(
    query:str,
    top_k: int  =5,
) -> list[str]:
    results = store.search(
        query = query,
        top_k = top_k,
    )

    return results["ids"][0]

def hybrid_retrieve(
    query: str,
    top_k: int = 5,
) -> list[str]:
    results = hybrid_retriever.search(
        query = query,
        top_k = top_k,
    )

    return[
        result["id"]
        for result in results
    ]

def reranked_retrieve(
    query: str,
    candidate_k: int = 5,
) -> list[str]:
    candidates = hybrid_retriever.search(
        query=query,
        top_k=candidate_k,
    )

    results = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=candidate_k,
    )

    return [
        result["id"]
        for result in results
    ]
pipelines = {
    "dense" : dense_retrieve,
    "hybrid": hybrid_retrieve,
    "hybrid_reranked": reranked_retrieve,
}

for pipeline_name, retrieve_function in pipelines.items():

    hit_1_scores = []
    hit_3_scores = []
    recall_3_scores = []
    reciprocal_ranks = []

    print("\n" + "=" * 80)
    print(
        "PIPELINE:",
        pipeline_name.upper(),
    )

    for example in evaluation_data:

        query = example["query"]

        relevant_ids =(
            example["relevant_chunk_ids"]
        )

        retrieved_ids = retrieve_function(query)
        hit_1 = hit_at_k(
            retrieved_ids,
            relevant_ids,
            k=1,
        )

        hit_3 = hit_at_k(
            retrieved_ids,
            relevant_ids,
            k=3,
        )

        recall_3 = recall_at_k(
            retrieved_ids,
            relevant_ids,
            k=3,
        )

        rr = reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        )

        hit_1_scores.append(hit_1)
        hit_3_scores.append(hit_3)
        recall_3_scores.append(recall_3)
        reciprocal_ranks.append(rr)

        print("\nQuery:", query)
        print("Expected:", relevant_ids)
        print("Retrieved:", retrieved_ids[:3])


    number_of_queries = len(evaluation_data)

    mean_hit_1 = (
        sum(hit_1_scores)/ number_of_queries
    )
    mean_hit_3 = (
     sum(hit_3_scores)
    / number_of_queries
    )

    mean_recall_3 = (
        sum(recall_3_scores)
        / number_of_queries
    )

    mrr = (
        sum(reciprocal_ranks)
        / number_of_queries
    )

    print("\n" + "-" * 80)

    print(
        f"Hit@1:   {mean_hit_1:.3f}"
    )

    print(
        f"Hit@3:   {mean_hit_3:.3f}"
    )

    print(
        f"Recall@3:{mean_recall_3:.3f}"
    )

    print(
        f"MRR:     {mrr:.3f}"
    )






