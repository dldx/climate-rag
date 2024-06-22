import argparse
import os
from typing import List, Literal, Optional
from dotenv import load_dotenv
from tools import get_vector_store
import langcodes


load_dotenv()  # take environment variables from .env.

import sys

# Different display functions for Jupyter and terminal
if "ipykernel" in sys.modules:
    from IPython.display import display, Markdown

    pretty_print = display
else:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.pretty import pprint

    console = Console()
    pretty_print = console.print


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    # query_text is only required if get_document is not provided.
    parser.add_argument(
        "query_text",
        type=str,
        help="The query text.",
        nargs="?" if "--get-documents" in sys.argv else 1,
    )
    parser.add_argument(
        "--rag-filter", type=str, help="Optional rag filter to use", default=""
    )
    # Add optional arguments.
    parser.add_argument(
        "--search-tool",
        type=str,
        help="The search tool to use. One of 'serper', 'tavily', 'baidu'",
        default="serper",
    )
    parser.add_argument(
        "--improve-question",
        action=argparse.BooleanOptionalAction,
        help="Whether to improve the question before querying the database",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        help="Whether to rerank the documents after retrieval",
    )
    parser.add_argument(
        "--crawl",
        action=argparse.BooleanOptionalAction,
        help="Whether to crawl the web for more documents",
    )
    parser.add_argument(
        "--initial-generation",
        action=argparse.BooleanOptionalAction,
        help="Whether to generate before querying the web",
    )

    parser.add_argument(
        "--language",
        type=str,
        help="The language to query in. One of 'en', 'zh', 'vi'",
        default="en",
    )

    parser.add_argument(
        "--llm",
        type=str,
        help="The language model to use. One of 'gpt-4o', 'mistral', 'claude'",
        default="claude",
    )

    parser.add_argument(
        "--get-documents",
        help="Get documents by their IDs",
        nargs="+",
        default=[],
    )

    parser.set_defaults(
        rerank=True, crawl=True, improve_question=True, initial_generation=True
    )

    args = parser.parse_args()
    query_text = args.query_text[0] if type(args.query_text) == list else args.query_text

    db = get_vector_store()
    if len(args.get_documents) > 0:
        get_documents_from_db(db, args.get_documents)
    else:
        run_query(
            query_text,
            db,
            llm=args.llm,
            rag_filter=args.rag_filter,
            improve_question=args.improve_question,
            search_tool=args.search_tool,
            do_rerank=args.rerank,
            do_crawl=args.crawl,
            language=args.language,
            initial_generation=args.initial_generation,
        )


def get_documents_from_db(db, doc_ids: List[str]):
    import pandas as pd

    df = (
        pd.DataFrame(db.get(ids=doc_ids))
        .dropna(axis=1, how="all")
        .drop(["ids"], axis=1)
        .to_dict(orient="records")
    )
    # Return df as list of rows
    for row in df:
        pretty_print(
            Markdown(
                f"""
---
# {row["metadatas"]}
{row["documents"]}
"""
            )
        )


def run_query(
    question: str,
    db,
    llm: Literal["gpt-4o", "mistral", "claude"] = "claude",
    rag_filter: Optional[str] = None,
    improve_question: Optional[bool] = True,
    search_tool: Literal["serper", "tavily", "baidu"] = "serper",
    do_rerank: Optional[bool] = True,
    do_crawl: Optional[bool] = True,
    language: Literal["en", "zh", "vi"] = "en",
    initial_generation: Optional[bool] = True,
):
    from graph import create_graph

    app = create_graph()

    # Run
    inputs = {
        "question": question,
        "question_en": question,
        "db": db,
        "llm": llm,
        "rag_filter": rag_filter,
        "shall_improve_question": improve_question,
        "search_tool": search_tool,
        "do_rerank": do_rerank,
        "language": language,
        "initial_generation": initial_generation,
    }
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            print(f"Node '{key}':")
            # Optional: print full state at each node
            if key == "generate":
                pretty_print(
                    Markdown(
                        f"""# {question}\n\n"""
                        + value["generation"]
                        + "\n\n**Sources:**\n\n"
                        + "\n\n".join(
                            set(
                                [
                                    (
                                        " * " + doc.metadata["source"]
                                        if "source" in doc.metadata.keys()
                                        else ""
                                    )
                                    for doc in value["documents"]
                                ]
                            )
                        )
                    )
                )
        print("\n---\n")

    return value


if __name__ == "__main__":
    main()
