from app.ingestion.text_loader import load_text_document
from app.ingestion.chunker import chunk_document_by_paragraphs
from app.retrieval.vector_store import LocalVectorStore

files = [
    "data/raw/hypertension_guideline.txt",
    "data/raw/diabetes_guideline.txt",
    "data/raw/heart_failure_guideline.txt",

]

all_chunks = []

for file_path in files:

    document = load_text_document(file_path)
    chunks = chunk_document_by_paragraphs(
        document,
        max_chunk_size=400,
    )
    all_chunks.extend(chunks)

vector_store = LocalVectorStore()
vector_store.build_index(all_chunks)

print(f" {len(all_chunks)} Vector index built and saved to data/processed/vector_index")
