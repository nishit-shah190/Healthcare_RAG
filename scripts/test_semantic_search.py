from app.ingestion.text_loader import load_text_document
from app.ingestion.chunker import chunk_document_by_paragraphs
from app.retrieval.semantic_search import semantic_search


files = [
    "data/raw/hypertension_guideline.txt",
    "data/raw/diabetes_guideline.txt",
    "data/raw/heart_failure_guideline.txt",
]

all_chunks = []

for file_path in files:
    doc = load_text_document(file_path)

    chunks = chunk_document_by_paragraphs(
        doc,
        max_chunk_size=400,
    )

    all_chunks.extend(chunks)


queries = [
    "What lifestyle changes help patients with high blood pressure?",
    "What should be considered when treating someone with diabetes and kidney problems?",
    "How should patients with worsening heart symptoms be monitored?",
]


for query in queries:
    print("\n" + "=" * 80)
    print("QUERY:", query)

    results = semantic_search(
        query=query,
        chunks=all_chunks,
        top_k=3,
    )

    for result in results:
        print("-" * 60)
        print("Score:", round(result["score"], 3))
        print("Document:", result["chunk"].document_id)
        print(result["chunk"].content)