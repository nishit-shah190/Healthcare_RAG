from app.ingestion.text_loader import load_text_document


def test_load_text_document():
    doc = load_text_document(
        "data/raw/hypertension_guideline.txt"
    )

    assert doc.document_id == "hypertension_guideline"
    assert doc.title == "Adult Hypertension Clinical Guideline"
    assert doc.organization == "Example Health Institute"
    assert doc.publication_year == 2024
    assert doc.section == "3.1"
    assert doc.document_type == "clinical_guideline"

    assert "Title:" not in doc.content
    assert "Organization:" not in doc.content

    assert doc.content.startswith(
        "Adults diagnosed with hypertension"
    )