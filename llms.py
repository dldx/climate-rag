import os
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv

load_dotenv()


def get_chatbot(
    llm: Literal[
        "gpt-4o", "gpt-4", "gpt-4o-mini", "gpt-3.5-turbo-16k", "mistral", "claude"
    ] = "claude",
    **kwargs
):
    """Get a chatbot instance.

    Args:
        llm: The language model to use. Options are "gpt-4o", "gpt-4", "gpt-4o-mini", "gpt-3.5-turbo-16k", "mistral", or "claude".
        **kwargs: optional keyword arguments to pass to the chat model

    Returns:
        A chatbot instance.
    """

    if llm in ["gpt-4o", "gpt-4", "gpt-4o-mini", "gpt-3.5-turbo-16k"]:
        return ChatOpenAI(model=llm, **kwargs)
    elif llm in ["llama-3.1"]:
        return ChatNVIDIA(
            model="meta/llama-3.1-405b-instruct",
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.2,
            top_p=0.7,
            max_tokens=4096,
        )
    elif llm == "mistral":
        return ChatOllama(model="mistral", temperature=0, **kwargs)
        # return OllamaFunctions(model="mistral", temperature=0, format="json", **kwargs)
    elif llm == "claude":
        return ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            timeout=None,
            max_retries=2,
        )
