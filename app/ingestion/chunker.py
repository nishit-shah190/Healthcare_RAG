from app.models import HealthcareDocument, DocumentChunk

def chunk_document(
    document: HealthcareDocument,
    chunk_size: int =200,
    overlap: int =50
) -> list[DocumentChunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = document.content
    chunks = []
    start=0
    while start < len(text):

        end = start + chunk_size

        chunk_text = text[start:end]

        chunk_index = len(chunks)

        chunk = DocumentChunk(
            chunk_id=f"{document.document_id}_chunk_{chunk_index}",
            document_id=document.document_id,
            content=chunk_text.strip(),
            chunk_index=chunk_index,
        )

        chunks.append(chunk)
        start = end - overlap

    return chunks

def chunk_document_by_paragraphs(
    document: HealthcareDocument,
    max_chunk_size: int = 400,
) -> list[DocumentChunk]:

    paragraphs = document.content.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        proposed_chunk = (
            current_chunk + "\n\n" + paragraph
            if current_chunk
            else paragraph
        )

        if len(proposed_chunk) <= max_chunk_size:
            current_chunk = proposed_chunk

        else:
            if current_chunk:
                chunk_index = len(chunks)

                chunk = DocumentChunk(
                    chunk_id=(
                        f"{document.document_id}_paragraph_chunk_"
                        f"{chunk_index}"
                    ),
                    document_id=document.document_id,
                    content=current_chunk,
                    chunk_index=chunk_index,
                     title=document.title,
                    organization = document.organization,\
                    publication_year = document.publication_year,
                    section=document.section,
                    document_type=document.document_type,
                )

                chunks.append(chunk)

            current_chunk = paragraph

    if current_chunk:
        chunk_index = len(chunks)

        chunk = DocumentChunk(
            chunk_id=(
                f"{document.document_id}_paragraph_chunk_"
                f"{chunk_index}"
            ),
            document_id=document.document_id,
            content=current_chunk,
            chunk_index=chunk_index,
            title=document.title,
            organization = document.organization,\
            publication_year = document.publication_year,
            section=document.section,
            document_type=document.document_type,
        )

        chunks.append(chunk)

    return chunks


