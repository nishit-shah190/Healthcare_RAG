from pathlib import Path

from app.models import HealthcareDocument


def load_text_document(file_path: str) -> HealthcareDocument:
    path = Path(file_path)

    content = path.read_text(encoding="utf-8")

    metadata_text, body = content.split("\n\n",1)

    metadata = {}

    for line in metadata_text.splitlines():
        line = line.strip()

        if line.startswith("Title:"):
            metadata["title"] = line.replace("Title:", "").strip()

        elif line.startswith("Organization:"):
            metadata["organization"] = line.replace("Organization:", "").strip()

        elif line.startswith("Publication Year:"):
            year = line.replace("Publication Year:", "").strip()
            metadata["publication_year"] = int(year)

        elif line.startswith("Section:"):
            metadata["section"] = line.replace("Section:", "").strip()

        elif line.startswith("Document Type:"):
            metadata["document_type"] = line.replace(
                "Document Type:", ""
            ).strip()

    document = HealthcareDocument(
        document_id=path.stem,
        title=metadata["title"],
        source=str(path),
        content=body.strip(),
        document_type=metadata["document_type"],
        organization=metadata["organization"],
        publication_year=metadata["publication_year"],
        section=metadata["section"],
    )

    return document