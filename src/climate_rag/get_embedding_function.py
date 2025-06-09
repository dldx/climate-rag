import os

from dotenv import load_dotenv

load_dotenv()


def get_embedding_function(model="text-embedding-ada-002"):
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    if model == "text-embedding-ada-002":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings()
    elif model == "qwen3-embeddings-4b":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model="qwen3-embeddings-4b",
            base_url=os.getenv("QWEN_EMBEDDINGS_BASE_URL"),
            api_key=os.getenv("QWEN_EMBEDDINGS_API_KEY"),
        )
    elif model == "gemini-embedding-001":
        from langchain_google_vertexai import VertexAIEmbeddings

        return VertexAIEmbeddings(model_name="gemini-embedding-001")
