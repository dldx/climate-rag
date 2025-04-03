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
    elif "voyage" in model:
        import chromadb.utils.embedding_functions as embedding_functions

        voyageai_ef = embedding_functions.VoyageAIEmbeddingFunction(
            api_key=os.getenv("VOYAGEAI_API_KEY"), model_name=model
        )

        return voyageai_ef
