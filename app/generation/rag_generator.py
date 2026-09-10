import os
from typing import Any,cast
import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
    http_client = cast(Any, httpx.Client())
)

MODEL_NAME = "gpt-5.6"

def build_context(
    retrieved_chunks: list[dict],
) -> str:
    context_parts = []

    for index, result in enumerate(
        retrieved_chunks,
        start=1,
    ):
        # Each retrieved result already contains text and metadata.
        metadata = result["metadata"]

        source_block = f"""
        SOURCE [{index}]
        Title: {metadata["title"]}
        Document ID: {metadata["document_id"]}
        Section: {metadata["section"]}
        Organization: {metadata["organization"]}
        Publication Year: {metadata["publication_year"]}
        Content:
        {result["document"]}
        """

        context_parts.append(
            source_block.strip()
        )

    return "\n\n".join(context_parts)
def build_prompt(
    query: str,
    context: str,
) -> str:
    # Explicit grounding rules reduce unsupported generation.
    return f"""
    You are a healthcare knowledge copilot.

    Answer the user's question using ONLY the provided sources.

    Rules:
    1. Do not use outside medical knowledge.
    2. If the sources do not contain enough information, say:
    "I do not have enough information in the retrieved knowledge base to answer this reliably."
    3. Do not invent diagnoses, medications, recommendations, or dosages.
    4. Cite supporting sources using [1], [2], etc.
    5. Keep the answer concise and factual.
    6. After the answer, include a "Sources" section listing the cited source numbers and titles.

    User question:
    {query}

    Retrieved sources:
    {context}
    """.strip()

def generate_answer(
    query:str,
    retrieved_chunks: list[dict],
) -> str:
    context = build_context(
        retrieved_chunks
    )

    prompt = build_prompt(
        query= query,
        context=context,
    )

    response = client.responses.create(
        model = MODEL_NAME,
        input=prompt,
    )

    return response.output_text

