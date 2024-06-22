from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv

load_dotenv()

def get_chatbot(llm: Literal["gpt-4o", "mistral", "claude"] = "claude", **kwargs):
    """Get a chatbot instance.

    Args:
        llm: The language model to use. Options are "gpt-4o", "mistral", or "claude".
        **kwargs: optional keyword arguments to pass to the chat model

    Returns:
        A chatbot instance.
    """

    if llm == "gpt-4o":
        return ChatOpenAI(model="gpt-4o", **kwargs)
    elif llm == "mistral":
        return ChatOllama(model="mistral", temperature=0, **kwargs)
    elif llm == "claude":
        return ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    timeout=None,
    max_retries=2,
)