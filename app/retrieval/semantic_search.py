import numpy as np

from app.models import DocumentChunk
from app.retrieval.embeddings import(
    embed_chunks,
    embed_text,
    cosine_similarity,
)

def semantic_search(
    query:str,
    chunks:list[DocumentChunk],
    top_k:int = 3,
):
    chunk_embeddings = embed_chunks(chunks)
    query_embedding = embed_text(query)

    scores =[]

    for chunk_embedding in chunk_embeddings:
        score = cosine_similarity(
            query_embedding,
            chunk_embedding,
        )

        scores.append(score)
    
    scores = np.array(scores)
    ranked_indices = np.argsort(scores)[::-1]
    results = []

    for i in ranked_indices[:top_k]:
        results.append(
            {
                "chunk": chunks[i],
                "score": float(scores[i]),
            }
        )
    return results