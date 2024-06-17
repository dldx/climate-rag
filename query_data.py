import argparse
import os
from typing import Literal, Optional
from dotenv import load_dotenv
from tools import get_vector_store
import langcodes


load_dotenv()  # take environment variables from .env.

import sys

# Different display functions for Jupyter and terminal
if "ipykernel" in sys.modules:
    from IPython.display import display, Markdown

    pprint = display
else:
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    pprint = console.print


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
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
        "--rerank",
        action=argparse.BooleanOptionalAction,
        help="Whether to rerank the documents after retrieval",
    )

    parser.add_argument(
        "--language",
        type=str,
        help="The language to query in. One of 'en', 'zh', 'vi'",
        default="en",
    )

    parser.set_defaults(rerank=True)

    args = parser.parse_args()
    query_text = args.query_text
    db = get_vector_store()
    run_query(
        query_text,
        db,
        rag_filter=args.rag_filter,
        search_tool=args.search_tool,
        do_rerank=args.rerank,
        language=args.language,
    )


def run_query(
    question: str,
    db,
    rag_filter: Optional[str] = None,
    improve_question: Optional[bool] = True,
    search_tool: Literal["serper", "tavily", "baidu"] = "serper",
    do_rerank: Optional[bool] = True,
    language: Literal["en", "zh", "vi"] = "en",
):
    from graph import create_graph

    app = create_graph()

    # Run
    inputs = {
        "question": question,
        "db": db,
        "rag_filter": rag_filter,
        "shall_improve_question": improve_question,
        "search_tool": search_tool,
        "do_rerank": do_rerank,
        "language": language,
    }
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            print(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            if key == "generate":
                pprint(
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
