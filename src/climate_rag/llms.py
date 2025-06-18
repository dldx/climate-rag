import os

from dotenv import load_dotenv

from climate_rag.schemas import SupportedLLMs

load_dotenv()


def get_chatbot(
    llm: SupportedLLMs,
    **kwargs,
):
    """Get a chatbot instance.

    Args:
        llm: The language model to use. Options are "gemini-2.5-flash", "gemini-2.5-pro", "gpt-4o", "gpt-4.1", "gpt-4o-mini", "mistral", "claude" or "llama-3.1".
        **kwargs: optional keyword arguments to pass to the chat model

    Returns:
        A chatbot instance.
    """

    if "gpt" in llm:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=llm, **kwargs)
    elif "llama" in llm:
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
    elif "gemini" in llm:
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
    elif "mistral" in llm:
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(model="mistral", temperature=0, **kwargs)
    elif "claude" in llm:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            timeout=None,
            max_retries=2,
        )
    else:
        raise ValueError(f"Unknown LLM: {llm}")


def get_max_token_length(llm: SupportedLLMs) -> int:
    if llm == "gpt-4o":
        return 100_000
    elif "gemini" in llm or "gpt-4.1" in llm:
        # Can handle 1 million tokens but takes too long to process so not worth it!
        return 500_000
    elif "o4-mini" in llm or "o3" in llm:
        return 150_000
    else:
        return 24_000
