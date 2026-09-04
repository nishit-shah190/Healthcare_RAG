from app.models import HealthcareDocument


def test_healthcare_document_creation():

    doc = HealthcareDocument(
        document_id="doc_001",
        title="Hypertension Guideline",
        source="synthetic",
        document_type="clinical_guideline",
        content="Example healthcare guideline.",
        organization= "baylor Scott",
        publication_year=2002,
        section = "3.1"
    )

    assert doc.document_id == "doc_001"
    assert doc.document_type == "clinical_guideline"