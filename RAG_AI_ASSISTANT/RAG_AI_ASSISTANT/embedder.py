from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List

# Load environment variables from .env file
load_dotenv()   
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536-dimension vectors


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Embeds chunks using OpenAI's embedding model."""
    embeddings = []
    for chunk in chunks:
        try:
            response = client.embeddings.create(
                input=chunk,
                model=EMBEDDING_MODEL
            )
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            print(f"Error embedding chunk: {e}")
            embeddings.append([])  # or handle differently
    return embeddings


def embed_User_query(query: str) -> List[float]:
    """Embeds a user query using OpenAI's embedding model."""
    try:
        response = client.embeddings.create(
            input=query,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error embedding user query: {e}")
        return []  # or raise




