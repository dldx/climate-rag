import argparse
import os
import sys
from typing import Iterator, List, Literal, Optional, Tuple

import pandas as pd
import redis
from dotenv import load_dotenv

from climate_rag.agents import GraphState
from climate_rag.cache import r
from climate_rag.constants import language_choices
from climate_rag.graph import create_graph
from climate_rag.helpers import clean_urls
from climate_rag.tools import (
    format_docs,
    get_sources_based_on_filter,
    get_vector_store,
)

# os.environ["LANGCHAIN_TRACING_V2"] = "true"


load_dotenv()  # take environment variables from .env.
rag_graph = create_graph()


# Different display functions for Jupyter and terminal
if "ipykernel" in sys.modules:
    from IPython.display import Markdown, display

    pretty_print = display
else:
    from rich.console import Console
    from rich.markdown import Markdown

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
        "--project-id",
        type=str,
        help="The project ID to query for.",
        default="langchain",
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
        help=f"The language to query in. Any two letter language code or name will work. Eg. {' '.join([f'{name} ({code})' for name, code in language_choices])}",
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
        "--print-docs",
        action=argparse.BooleanOptionalAction,
        help="Whether to print a formatted list of documents returned by the query",
    )

    parser.add_argument(
        "--print-docs-only",
        action=argparse.BooleanOptionalAction,
        help="Whether to print only the documents returned by the query and then exit.",
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
        action="store_true",
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
        args.query_text[0] if type(args.query_text) is list else args.query_text
    )

    if args.language in ("vi", "th"):
        if args.rerank is None:
            args.rerank = False
    else:
        if args.rerank is None:
            args.rerank = True

    db = get_vector_store(args.project_id)
    if len(args.get_docs) > 0:
        get_documents_from_db(db, args.get_docs)
    elif args.get_source_doc is not None:
        query_source_documents(
            db,
            args.get_source_doc,
            fields=["source", "page_content"],
            project_id=args.project_id,
        )
    else:
        # If query_text is too short, return a message
        if len(query_text) < 5:
            pretty_print(
                "Please provide a more complete question of at least 5 characters."
            )
            return
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
                project_id="langchain",
            ):
                if (
                    args.add_additional_metadata and key == "add_additional_metadata"
                ) or (
                    (args.add_additional_metadata is False)
                    and key == "retrieve_from_database"
                ):
                    documents = value["documents"]
                    if args.print_docs:
                        print(format_docs(documents))
                    if args.print_docs_only:
                        print(format_docs(documents))
                        sys.exit(0)

            # Ask user for feedback
            if key == "__interrupt__":
                if args.yes:
                    user_happy_with_answer = True
                else:
                    user_happy_with_answer = (
                        input(
                            "Are you happy with the answer? (y / n=web search)"
                        ).lower()[0]
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
    limit: Optional[int] = 10_000,
    project_id: Optional[str] = "langchain",
) -> pd.DataFrame:
    """
    Query the database for documents with a specific metadata key-value pair. Currently only supports querying by source.

    Args:
        db: The database object
        source_uri: The source URI to query for. eg. "*carbontracker.org*" for wildcard search or "https://carbontracker.org" for exact match
        print: Whether to print the results
        fields: The fields to return in the DataFrame. Defaults to all fields. Available fields are: "source", "page_content", "raw_html", "date_added", "page_length", "title", "company_name"
        project_id: The project ID to query for. Defaults to "langchain"

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
    keys = (
        (
            f"climate-rag::{project_id}::source:"
            if project_id != "langchain"
            else "climate-rag::source:"
        )
        + pd.Series(
            get_sources_based_on_filter(
                rag_filter=source_uri, limit=limit, r=r, project_id=project_id
            )
        )
    ).tolist()
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
        df = pd.concat(all_docs, axis=1).T
        if "date_added" in fields:
            df = df.assign(
                # Convert unix timestamp to datetime
                date_added=lambda x: pd.to_datetime(
                    x["date_added"].astype(float), unit="s", errors="coerce"
                ).dt.strftime("%Y-%m-%d"),
            )
        df = df.reindex(columns=fields)
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
    llm: Literal[
        "gpt-4o",
        "gpt-4o-mini",
        "mistral",
        "claude",
        "llama-3.1",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ] = "claude",
    rag_filter: Optional[str] = None,
    improve_question: Optional[bool] = True,
    search_tool: Literal["serper", "tavily", "baidu"] = "serper",
    do_rerank: Optional[bool] = True,
    do_crawl: Optional[bool] = True,
    max_search_queries: Optional[int] = 1,
    do_add_additional_metadata: Optional[bool] = True,
    language: Literal[
        "en", "ar", "zh", "de", "id", "it", "ja", "kk", "ko", "ru", "es", "vi", "tl"
    ] = "en",
    initial_generation: Optional[bool] = True,
    history: Optional[List] = [],
    mode: Optional[Literal["gui", "cli"]] = "cli",
    thread_id: Optional[str] = "1",
    happy_with_answer: Optional[bool] = False,
    continue_after_interrupt: Optional[bool] = False,
    project_id: Optional[str] = "langchain",
) -> Iterator[Tuple[str, GraphState]]:
    if len(question) < 5:
        return "error", {"error": "Please provide a more complete question."}

    if len(language) > 2:
        # Convert language name to langcode
        import langcodes

        language = langcodes.find(language).language

    # Run
    inputs = {
        "question": question,
        "question_en": question,
        "project_id": project_id,
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
        rag_graph.update_state(
            thread,
            {
                "project_id": project_id,
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
    for output in rag_graph.stream(
        None if continue_after_interrupt else inputs, thread
    ):
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
