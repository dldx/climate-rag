import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()


def get_embedding_function(
    model: Literal[
        "text-embedding-ada-002",
        "gemini-embedding-001",
    ]
    | str = "text-embedding-ada-002",
):
    if model == "text-embedding-ada-002":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings()
    elif model == "gemini-embedding-001":
        from langchain_google_vertexai import VertexAIEmbeddings

        return VertexAIEmbeddings(model_name="gemini-embedding-001")
    elif len(model) > 0:
        return OpenAICompatibleEmbeddings(model=model)
    else:
        raise ValueError(f"Invalid model: {model}")


class OpenAICompatibleEmbeddings(Embeddings):
    """
    LangChain compatible embedding class for OpenAI-compatible APIs.
    """

    def __init__(self, model: str):
        """
        Initialize the OpenAI compatible embeddings.

        Args:
            model: The embedding model to use
        """
        from openai import OpenAI

        if os.getenv("OPENAI_COMPATIBLE_EMBEDDINGS_BASE_URL") is None:
            raise ValueError("OPENAI_COMPATIBLE_EMBEDDINGS_BASE_URL is not set")
        if os.getenv("OPENAI_COMPATIBLE_EMBEDDINGS_API_KEY") is None:
            raise ValueError("OPENAI_COMPATIBLE_EMBEDDINGS_API_KEY is not set")

        self.model = model
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_COMPATIBLE_EMBEDDINGS_BASE_URL"),
            api_key=os.getenv("OPENAI_COMPATIBLE_EMBEDDINGS_API_KEY"),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        response = self.client.embeddings.create(
            model=self.model, input=texts, encoding_format="float"
        )
        return [embedding.embedding for embedding in response.data]

    def embed_query(self, text: str) -> list[float]:
        """
        Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        response = self.client.embeddings.create(
            model=self.model, input=[text], encoding_format="float"
        )
        return response.data[0].embedding
