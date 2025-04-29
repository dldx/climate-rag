import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


def get_chatbot(
    llm: Literal[
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gpt-4o",
        "gpt-4",
        "gpt-4o-mini",
        "gpt-3.5-turbo-16k",
        "mistral",
        "claude",
        "llama-3.1",
    ] = "claude",
    **kwargs,
):
    """Get a chatbot instance.

    Args:
        llm: The language model to use. Options are "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gpt-4o", "gpt-4", "gpt-4o-mini", "gpt-3.5-turbo-16k", "mistral", "claude" or "llama-3.1".
        **kwargs: optional keyword arguments to pass to the chat model

    Returns:
        A chatbot instance.
    """

    if llm in ["gpt-4o", "gpt-4", "gpt-4o-mini", "gpt-3.5-turbo-16k"]:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=llm, **kwargs)
    elif llm in ["llama-3.1"]:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        if "NVIDIA_API_KEY" not in os.environ:
            raise ValueError(
                "`NVIDIA_API_KEY` is needed to call llama-3.1 but it is not set in .env."
            )
        return ChatNVIDIA(
            model="meta/llama-3.1-405b-instruct",
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.2,
            top_p=0.7,
            max_tokens=4096,
        )
    elif llm in ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"]:
        from langchain_google_vertexai import ChatVertexAI

        # The GOOGLE_PROJECT_ID environment variable must be set
        if "GOOGLE_PROJECT_ID" not in os.environ or not os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        ):
            raise ValueError(
                "`GOOGLE_PROJECT_ID` and `GOOGLE_APPLICATION_CREDENTIALS` are both needed to call Gemini but are not set in .env. See https://cloud.google.com/docs/authentication/provide-credentials-adc#how-to for details"
            )

        return ChatVertexAI(
            model=llm,
            project=os.getenv("GOOGLE_PROJECT_ID"),
            **kwargs,
        )
    elif llm == "mistral":
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(model="mistral", temperature=0, **kwargs)
    elif llm == "claude":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            timeout=None,
            max_retries=2,
        )


def get_max_token_length(llm: str) -> int:
    if llm == "gpt-4o":
        return 24_000
    elif llm in ("gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"):
        # Can handle 1 million tokens but takes too long to process so not worth it!
        return 500_000
    else:
        return 24_000
