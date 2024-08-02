from langgraph.graph import END, StateGraph
from agents import (
    improve_question,
    formulate_query,
    generate_search_query,
    retrieve,
    add_additional_metadata,
    grade_documents,
    rerank_docs,
    generate,
    ask_user_for_feedback,
    web_search,
    add_urls_to_database,
    decide_to_rerank,
    decide_to_search,
    decide_to_generate,
    decide_to_add_additional_metadata,
    GraphState
)
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

def create_graph():

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("improve_question", improve_question)  # transform_query
    workflow.add_node("formulate_query", formulate_query)  # formulate query
    # workflow.add_node("generate_search_query", generate_search_query)  # generate search query
    workflow.add_node("retrieve_from_database", retrieve)  # retrieve
    workflow.add_node("add_additional_metadata", add_additional_metadata)  # add additional metadata
    workflow.add_node("rerank_documents", rerank_docs)  # rerank documents
    # workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generate
    workflow.add_node("ask_user_for_feedback", ask_user_for_feedback)  # ask user for feedback
    workflow.add_node("web_search_node", web_search)  # web search
    workflow.add_node("add_urls_to_database", add_urls_to_database)  # add new urls to db

    # Build graph
    workflow.set_entry_point("improve_question")
    workflow.add_edge("improve_question", "formulate_query")
    # workflow.add_edge("formulate_query", "generate_search_query")
    workflow.add_conditional_edges("formulate_query",
                                  decide_to_generate,
                                  {
                                    "generate":  "retrieve_from_database",
                                    "no_generate": "web_search_node"})
    workflow.add_edge("retrieve_from_database", "add_additional_metadata")
    workflow.add_conditional_edges(
        "add_additional_metadata",
        decide_to_rerank,
        {
            "rerank": "rerank_documents",
            "no_rerank": "generate"
        }
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
    app = workflow.compile()

    return app