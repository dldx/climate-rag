import pickle
from typing import Any, Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.graph import END, StateGraph

from climate_rag.agents import (
    add_additional_metadata,
    add_urls_to_database,
    ask_user_for_feedback,
    decide_to_generate,
    decide_to_rerank,
    decide_to_search,
    formulate_query,
    generate,
    improve_question,
    rerank_docs,
    retrieve,
    web_search,
)
from climate_rag.schemas import GraphState

load_dotenv()


class CustomSerializer(SerializerProtocol):
    """Custom serializer that handles pandas DataFrames, chromadb Documents and other objects."""

    def dumps(self, obj: Any) -> bytes:
        if isinstance(obj, pd.DataFrame):
            # For DataFrames, we'll pickle them with a special type marker
            return pickle.dumps(("DataFrame", obj.to_dict()))
        elif isinstance(obj, Document):
            # For chromadb Documents, we'll pickle them with a special type marker
            return pickle.dumps(("Document", obj))
        return pickle.dumps(obj)

    def dumps_typed(self, obj: Any) -> Tuple[str, bytes]:
        if isinstance(obj, pd.DataFrame):
            # For DataFrames, we'll pickle them with a special type marker
            return "DataFrame", pickle.dumps(obj.to_dict())
        elif isinstance(obj, Document):
            # For chromadb Documents, we'll pickle them with a special type marker
            return "Document", pickle.dumps(obj)
        return "pickle", pickle.dumps(obj)

    def loads(self, data: bytes) -> Any:
        obj = pickle.loads(data)
        if isinstance(obj, tuple):
            if obj[0] == "DataFrame":
                # If it's our special DataFrame format, reconstruct it
                return pd.DataFrame.from_dict(obj[1])
            elif obj[0] == "Document":
                # If it's our special Document format, reconstruct it
                return obj[1]
        return obj

    def loads_typed(self, data: Tuple[str, bytes]) -> Any:
        type_str, bytes_data = data
        if type_str == "DataFrame":
            # If it's a DataFrame type, reconstruct it
            return pd.DataFrame.from_dict(pickle.loads(bytes_data))
        elif type_str == "Document":
            # If it's a Document type, reconstruct it
            doc_data = pickle.loads(bytes_data)
            return doc_data
        return pickle.loads(bytes_data)


# Create an instance of our custom serializer
serializer = CustomSerializer()


def create_graph():
    workflow = StateGraph(GraphState)
    memory = MemorySaver(serde=serializer)

    # Define the nodes
    workflow.add_node("improve_question", improve_question)  # transform_query
    workflow.add_node("formulate_query", formulate_query)  # formulate query
    # workflow.add_node("generate_search_query", generate_search_query)  # generate search query
    workflow.add_node("retrieve_from_database", retrieve)  # retrieve
    workflow.add_node(
        "add_additional_metadata", add_additional_metadata
    )  # add additional metadata
    workflow.add_node("rerank_documents", rerank_docs)  # rerank documents
    # workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generate
    workflow.add_node(
        "ask_user_for_feedback", ask_user_for_feedback
    )  # ask user for feedback
    workflow.add_node("web_search_node", web_search)  # web search
    workflow.add_node(
        "add_urls_to_database", add_urls_to_database
    )  # add new urls to db

    # Build graph
    workflow.set_entry_point("improve_question")
    workflow.add_edge("improve_question", "formulate_query")
    # workflow.add_edge("formulate_query", "generate_search_query")
    workflow.add_conditional_edges(
        "formulate_query",
        decide_to_generate,
        {"generate": "retrieve_from_database", "no_generate": "web_search_node"},
    )
    workflow.add_edge("retrieve_from_database", "add_additional_metadata")
    workflow.add_conditional_edges(
        "add_additional_metadata",
        decide_to_rerank,
        {"rerank": "rerank_documents", "no_rerank": "generate"},
    )
    workflow.add_edge("rerank_documents", "generate")
    workflow.add_edge("generate", "ask_user_for_feedback")
    workflow.add_edge("web_search_node", "add_urls_to_database")
    workflow.add_edge("add_urls_to_database", "retrieve_from_database")
    workflow.add_conditional_edges(
        "ask_user_for_feedback",
        decide_to_search,
        {
            "web_search": "web_search_node",
            "END": END,
        },
    )

    # Compile
    app = workflow.compile(checkpointer=memory, interrupt_after=["generate"])

    return app
