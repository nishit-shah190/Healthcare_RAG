from app.evaluation.retrieval_metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank
)

def test_hit_at_k():
    retrieved = ["A", "B", "C"]
    relevant = ["B"]

    assert hit_at_k(retrieved, relevant, 1) == 0.0
    assert hit_at_k(retrieved, relevant, 3) ==1.0

def test_recall_at_k():
    retrieved = ["A", "C", "D"]
    relevant = ["A", "B"]

    # We found one of the two relevant chunks.
    assert recall_at_k(retrieved, relevant, 3) == 0.5
def test_reciprocal_rank():
    retrieved = ["A", "B", "C"]
    relevant = ["B"]

    # Relevant chunk appears at rank 2 → reciprocal rank = 1/2.
    assert reciprocal_rank(retrieved, relevant) == 0.5