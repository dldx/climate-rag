import argparse
from graph import create_graph
import redis
import os
from typing import Iterator, List, Literal, Optional, Tuple
from dotenv import load_dotenv
import pandas as pd
from agents import GraphState
from helpers import clean_urls
from tools import get_vector_store
import langcodes
from cache import r
os.environ["LANGCHAIN_TRACING_V2"] = "true"


load_dotenv()  # take environment variables from .env.
app = create_graph()

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
        nargs="?" if "--get-docs" in sys.argv or "--get-source-doc" in sys.argv else 1,
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
        "--max-search-queries",
        type=int,
        help="The maximum number of search queries to run",
        default=1,
    )

    parser.add_argument(
        "--llm",
        type=str,
        help="The language model to use. One of 'gpt-4o', 'mistral', 'claude'",
        default="claude",
    )

    parser.add_argument(
        "--get-docs",
        help="Get documents by their IDs",
        nargs="+",
        default=[],
    )

    parser.add_argument(
        "--get-source-doc",
        help="Get documents by their source",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--add-additional-metadata",
        action=argparse.BooleanOptionalAction,
        help="Whether to add additional metadata to the documents to improve reranking",
    )

    parser.add_argument(
        "--yes",
        action='store_true',
        help="Whether to automatically accept the answer",
    )

    parser.set_defaults(
        crawl=True,
        improve_question=True,
        initial_generation=True,
        add_additional_metadata=True,
        yes=False,
    )

    args = parser.parse_args()
    query_text = (
        args.query_text[0] if type(args.query_text) == list else args.query_text
    )

    if args.language == "vi":
        if args.rerank is None:
            args.rerank = False
    else:
        if args.rerank is None:
            args.rerank = True

    db = get_vector_store()
    if len(args.get_docs) > 0:
        get_documents_from_db(db, args.get_docs)
    elif args.get_source_doc is not None:
        query_source_documents(db, args.get_source_doc, fields=["source", "page_content"])
    else:
        continue_after_interrupt = False
        user_happy_with_answer = False
        while user_happy_with_answer is False:
            for key, value in run_query(
                query_text,
                llm=args.llm,
                rag_filter=args.rag_filter,
                improve_question=args.improve_question,
                search_tool=args.search_tool,
                do_rerank=args.rerank,
                do_crawl=args.crawl,
                max_search_queries=args.max_search_queries,
                do_add_additional_metadata=args.add_additional_metadata,
                language=args.language,
                initial_generation=args.initial_generation,
                continue_after_interrupt=continue_after_interrupt,
                happy_with_answer=user_happy_with_answer,
            ):
                pass

            # Ask user for feedback
            if key == "generate":
                if args.yes:
                    user_happy_with_answer = True
                else:
                    user_happy_with_answer = (
                        input("Are you happy with the answer? (y / n=web search)").lower()[
                            0
                        ]
                        == "y"
                    )

                continue_after_interrupt = True


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


def query_source_documents(
    db,
    source_uri: str,
    print_output=True,
    fields: Optional[
        List[
            Literal[
                "source",
                "page_content",
                "raw_html",
                "date_added",
                "page_length",
                "title",
                "company_name",
            ]
        ]
    ] = None,
) -> pd.DataFrame:
    """
    Query the database for documents with a specific metadata key-value pair. Currently only supports querying by source.

    Args:
        db: The database object
        source_uri: The source URI to query for. eg. "*carbontracker.org*" for wildcard search or "https://carbontracker.org" for exact match
        print: Whether to print the results

    Returns:
        pd.DataFrame: The documents that match the query
    """
    if fields is None:
        fields = [
            "source",
            "page_content",
            "raw_html",
            "date_added",
            "page_length",
            "title",
            "company_name",
        ]
    keys = r.keys("climate-rag::source:" + source_uri)
    if len(keys) == 0:
        df = pd.DataFrame(columns=fields)
    else:
        all_docs = []
        for key in keys:
            try:
                doc = pd.Series(dict(zip(fields, r.hmget(key, *fields))))
            except redis.exceptions.ResponseError as e:
                print(e, key)

            all_docs.append(doc)
        df = (
            pd.concat(all_docs, axis=1)
            .T)
        if "date_added" in fields:
            df = (df.assign(
                # Convert unix timestamp to datetime
                date_added=lambda x: pd.to_datetime(
                    x["date_added"].astype(float), unit="s", errors="coerce"
                ).dt.strftime("%Y-%m-%d"),
            ))
        df = (df
            .reindex(
                columns=fields
            )
        )
    if print_output:
        # Return df as list of rows
        for row in df.to_dict(orient="records"):
            pretty_print(
                Markdown(
                    f"""
    ---
    # {row["source"]}
    {row["page_content"]}
    """
                )
            )

    return df


def get_all_documents_as_df(db) -> pd.DataFrame:
    import pandas as pd

    raw_data_df = (
        pd.DataFrame.from_records(db.get()["metadatas"])
        .assign(chunk_no=lambda x: x["id"].str.split(":").str[-1].astype(int))
        .reset_index()
        .sort_values(["source", "chunk_no"], ascending=True)
        .set_index("source")[["index", "id", "chunk_no", "date_added"]]
        .join(pd.DataFrame(dict(page_content=db.get()["documents"])), on="index")
        .drop(columns=["index", "id", "chunk_no"])
        .reset_index()
        .groupby("source")[["page_content", "date_added"]]
        .agg({"page_content": lambda x: "\n\n".join(x), "date_added": "first"})
        .reset_index()
        .assign(
            date_added=lambda x: (
                x["date_added"].astype("datetime64[ns]").astype(int) / 1e9
            ).astype(int)
        )
        .assign(page_length=lambda x: x["page_content"].str.len())
    )
    return raw_data_df


def run_query(
    question: str,
    llm: Literal["gpt-4o", "gpt-3.5-turbo-16k", "mistral", "claude"] = "claude",
    rag_filter: Optional[str] = None,
    improve_question: Optional[bool] = True,
    search_tool: Literal["serper", "tavily", "baidu"] = "serper",
    do_rerank: Optional[bool] = True,
    do_crawl: Optional[bool] = True,
    max_search_queries: Optional[int] = 1,
    do_add_additional_metadata: Optional[bool] = True,
    language: Literal["en", "zh", "vi"] = "en",
    initial_generation: Optional[bool] = True,
    history: Optional[List] = [],
    mode: Optional[Literal["gui", "cli"]] = "cli",
    thread_id: Optional[str] = "1",
    happy_with_answer: Optional[bool] = False,
    continue_after_interrupt: Optional[bool] = False,
) -> Iterator[Tuple[str, GraphState]]:

    # Run
    inputs = {
        "question": question,
        "question_en": question,
        "llm": llm,
        "rag_filter": rag_filter,
        "shall_improve_question": improve_question,
        "search_tool": search_tool,
        "do_rerank": do_rerank,
        "max_search_queries": max_search_queries,
        "do_add_additional_metadata": do_add_additional_metadata,
        "language": language,
        "initial_generation": initial_generation,
        "history": history,
        "mode": mode,
    }

    thread = {"configurable": {"thread_id": thread_id}}

    if continue_after_interrupt:
        app.update_state(
            thread,
            {
                "happy_with_answer": happy_with_answer,
                "do_rerank": do_rerank,
                "do_crawl": do_crawl,
                "do_add_additional_metadata": do_add_additional_metadata,
                "language": language,
                "shall_improve_question": improve_question,
                "max_search_queries": max_search_queries,
                "rag_filter": rag_filter,
            },
        )
    for output in app.stream(None if continue_after_interrupt else inputs, thread):
        for key, value in output.items():
            # Node
            print(f"Node '{key}':")
            # Optional: print full state at each node
            if key == "generate":
                pretty_print(
                    Markdown(
                        f"""# {value["initial_question"]}\n\n"""
                        + value["generation"]
                        + "\n\n**Sources:**\n\n"
                        + "\n\n".join(
                            set(
                                [
                                    (
                                        " * "
                                        + clean_urls(
                                            [doc.metadata["source"]],
                                            os.environ.get("STATIC_PATH", ""),
                                        )[0]
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

        yield key, value


if __name__ == "__main__":
    main()
