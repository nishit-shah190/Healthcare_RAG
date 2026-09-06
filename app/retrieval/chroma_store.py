import chromadb
from openai import embeddings, organization
from app.models import DocumentChunk
from app.retrieval.embeddings import (
    embed_chunks,
    embed_text
)

class ChromaVectorStore:

    def __init__(
        self,
        persist_dictionary: str = "data/processed/chroma_db",
        collection_name: str = "healthcare_guidelines",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_dictionary
        )

        self.collection = (
            self.client.get_or_create_collection( name = collection_name)
        )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        embeddings = embed_chunks(chunks)

        ids = []
        metadatas = []
        documents= []

        for chunk in chunks:
            ids.append(chunk.chunk_id)

            documents.append(chunk.content)

            metadatas.append({
                "document_id": chunk.document_id,
                "title": chunk.title,
                "organization": chunk.organization,
                "publication_year": chunk.publication_year,
                "secton" : chunk.section,
                "document_type": chunk.document_type,
                "chunk_index": chunk.chunk_index,
            })

        self.collection.upsert(
            ids = ids,
            documents = documents,
            embeddings= embeddings.tolist(),
            metadatas = metadatas,
        )
    def search(
        self,
        query:str,
        top_k : int = 3,
        where : dict | None = None,
    ):
        query_embedding = embed_text(
            query
        )

        results = self.collection.query(
            query_embeddings = [
                query_embedding.tolist()
            ],
            n_results= top_k,
            where = where,
        )

        return results