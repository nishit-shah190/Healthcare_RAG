import pytest

from app.ingestion.text_loader import load_text_document
from app.ingestion.chunker import chunk_document
from app.ingestion.chunker import chunk_document_by_paragraphs

def test_chunk_document():
    doc = load_text_document(
         "data/raw/hypertension_guideline.txt"
    )

    chunks = chunk_document(
        doc,
        chunk_size=200,
        overlap=50
    )

    assert len(chunks) > 1
    assert chunks[0].document_id == "hypertension_guideline"
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_id == "hypertension_guideline_chunk_0"


def test_overlap_must_be_smaller_than_chunk_size():

    doc = load_text_document(
        "data/raw/hypertension_guideline.txt"
    )

    with pytest.raises(ValueError):
        chunk_document(
            doc,
            chunk_size=100,
            overlap=100
        )
def test_paragraph_chunking():

    doc = load_text_document(
        "data/raw/hypertension_guideline.txt"
    )

    chunks = chunk_document_by_paragraphs(
        doc,
        max_chunk_size=400
    )

    assert len(chunks) > 0

    assert chunks[0].document_id == "hypertension_guideline"

    assert chunks[0].chunk_index == 0

    assert len(chunks[0].content) <= 400