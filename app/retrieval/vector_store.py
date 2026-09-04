from pathlib import Path
import json

import numpy as np

from app.models import DocumentChunk
from app.retrieval.embeddings import embed_chunks, embed_text

class LocalVectorStore:
    def __init__(
        self,
        index_dir: str = "data/processed/vector_index",
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = (
            self.index_dir / "embeddings.npy"
        )
        self.chunks_path =(
            self.index_dir / "chunks.json"
        )
    
    def build_index(self, chunks: list[DocumentChunk]) -> None:
        embeddings = embed_chunks(chunks)
        np.save(
            self.embeddings_path,
            embeddings,
        )
        chunk_data = [
            chunk.model_dump()
            for chunk in chunks
        ]
        with open(
            self.chunks_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(chunk_data, file, indent=2)

    def load_index(self):
        embeddings = np.load(self.embeddings_path)
        with open(
            self.chunks_path,
            "r",
            encoding="utf-8",
        ) as file:
            chunk_data = json.load(file)

        chunks =[
            DocumentChunk(**item)
            for item in chunk_data
        ]
        return embeddings, chunks

    def search(self, query: str, top_k:int =3,):
        embeddings, chunks = (
            self.load_index()
        )

        query_embedding = embed_text(query)

        scores = (
            embeddings @ query_embedding
        )

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