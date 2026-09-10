import json
from pathlib import Path

from app.retrieval import hybrid_retriever
from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.generation.rag_generator import generate_answer


from app.evaluation.generation_metrics import (
    fact_coverage,
    abstention_correct,
    citation_presence_score
)

eval_path = Path("app/evaluation/generation_eval.json")

with open(
    eval_path,
    "r",
    encoding ='utf-8',
) as file:
    eval_data = json.load(file)

store = ChromaVectorStore()

hybrid_retriever = HybridRetriever(chroma_store=store)

reranker = Reranker()

def run_rag(query: str) -> str:

    # Retrieve a reasonably broad candidate set first.
    candidates = hybrid_retriever.search(
        query=query,
        top_k=5,
    )

    # Improve final ordering before generation.
    retrieved_chunks = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=3,
    )

    # Generate only from the retrieved evidence.
    return generate_answer(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )

fact_scores = []
abstention_scores = []
citation_scores = []


for example in eval_data:

    query = example["query"]
    expected_facts = example["expected_facts"]
    should_abstain = example["should_abstain"]

    answer = run_rag(query)

    fact_score = fact_coverage(
        answer,
        expected_facts,
    )

    abstention_score = abstention_correct(
        answer,
        should_abstain,
    )

    citation_score = citation_presence_score(
        answer,
        should_abstain,
    )
    if not should_abstain:
        fact_scores.append(fact_score)
    abstention_scores.append(
        abstention_score
    )
    citation_scores.append(
        citation_score
    )

    print("\n" + "=" * 80)

    print(f"QUERY:\n{query}")

    print(
        f"\nSHOULD ABSTAIN: "
        f"{should_abstain}"
    )

    print(
        f"\nEXPECTED FACTS:\n"
        f"{expected_facts}"
    )

    print(f"\nANSWER:\n{answer}")

    print(
        f"\nFACT COVERAGE: "
        f"{fact_score:.3f}"
    )

    print(
        f"ABSTENTION CORRECT: "
        f"{abstention_score:.3f}"
    )

    print(
        f"CITATION PRESENT: "
        f"{citation_score:.3f}"
    )
number_of_examples = len(eval_data)


mean_fact_coverage = (
    sum(fact_scores)
    / len(fact_scores)
)

mean_abstention_accuracy = (
    sum(abstention_scores)
    / number_of_examples
)

mean_citation_presence = (
    sum(citation_scores)
    / number_of_examples
)


print("\n" + "=" * 80)
print("GENERATION EVALUATION SUMMARY")

print(
    f"Fact Coverage:        "
    f"{mean_fact_coverage:.3f}"
)

print(
    f"Abstention Accuracy: "
    f"{mean_abstention_accuracy:.3f}"
)

print(
    f"Citation Presence:   "
    f"{mean_citation_presence:.3f}"
)