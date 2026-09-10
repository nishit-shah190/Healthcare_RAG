def hit_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    top_k = retrieved_ids[:k]
    return float(
        any(
            chunk_id in relevant_ids
            for chunk_id in top_k
        )
    )

def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k:int,
) -> float:
    retrieved_top_k = set(
        retrieved_ids[:k]
    )

    relevant_set = set(
        relevant_ids    
    )
    matched = ( retrieved_top_k & relevant_set)

    return len(matched)/len(relevant_set)

def reciprocal_rank(
    retrieved_ids : list[str],
    relevant_ids: list[str],
) -> float:

    for rank, chunk_id in enumerate(
        retrieved_ids, start =1
    ):
        if chunk_id in relevant_ids:
            return 1.0/rank
    
    return 0.0





