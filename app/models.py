from pydantic import BaseModel, Field

class HealthcareDocument(BaseModel):
    document_id:str
    title: str
    source:str
    content:str

    document_type: str = Field(
        description= "Type of Healthcare docuemnt, such as guideline or policy"
    )
    organization: str
    publication_year: int
    section: str


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int


class IndexedChunk(BaseModel):
    chunk_id: str
    document_id:str
    content:str
    chunk_index:int