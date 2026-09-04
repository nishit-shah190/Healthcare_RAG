import os
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
from app.models import DocumentChunk


load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = "text-embedding-3-small"

def embed_text(text:str) -> np.ndarray:
    response = client.embeddings.create(
        model=MODEL_NAME,
        input=text
    )

    return np.array(
        response.data[0].embedding,
        dtype = np.float32,
    )

def embed_chunks(
    chunks: list[DocumentChunk]) -> np.ndarray:
        texts =[
            chunk.content
            for chunk in chunks
        ]

        response = client.embeddings.create(
            model=MODEL_NAME,
            input=texts,
        )

        embeddings = [
            item.embedding
            for item in response.data
        ]

        return np.array(
            embeddings,
            dtype = np.float32,
        )

def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    numerator = np.dot(
        vector_a,
        vector_b,
    )

    denominator = (
        np.linalg.norm(vector_a)*np.linalg.norm(vector_b)
    )

    return float(numerator/denominator)


