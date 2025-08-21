#!/usr/bin/env python3
"""
Database cleanup utility for removing problematic documents and short content.
"""

import argparse
import logging
import time
from typing import List

from redis.commands.search.document import Document
from redis.commands.search.query import Query

from climate_rag.cache import r, source_index_name
from climate_rag.tools import (
    delete_document_from_db,
    error_messages,
    get_vector_store,
    initialize_project_indices,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def remove_urls(urls: List[str], db) -> None:
    """
    Remove specified URLs from the database.

    Args:
        urls: List of URLs to remove
        db: Vector store database instance
    """
    for url in urls:
        delete_document_from_db(url, db, r)


def find_error_documents(project_id: str = "langchain") -> List[Document]:
    """
    Search for documents containing known error messages.

    Args:
        project_id: The project ID to query for

    Returns:
        List of document records containing errors
    """
    error_docs = []
    for error_message, response_message in error_messages.items():
        # Sanitize error message for search
        sanitized_message = (
            error_message.replace(":", "?").replace(",", "?").replace(".", "?")
        )

        # Search for documents with error message
        results = (
            r.ft(f"{source_index_name}_{project_id}")
            .search(
                Query(
                    f'@page_content: "{sanitized_message}"' """@page_length:[0 10000]"""
                )
                .dialect(2)
                .return_fields("source")
                .paging(0, 10000)
                .timeout(5000)
            )
            .docs
        )

        if results:
            logger.info(
                f"Found {len(results)} documents with error: {response_message}"
            )
            error_docs.extend(results)

    return error_docs


def find_short_documents(
    min_length: int = 400, project_id: str = "langchain"
) -> List[Document]:
    """
    Find documents shorter than specified length.

    Args:
        min_length: Minimum acceptable document length in characters
        project_id: The project ID to query for
    Returns:
        List of document records that are too short
    """
    return (
        r.ft(f"{source_index_name}_{project_id}")
        .search(
            Query(f"@page_length:[0 {min_length}]").dialect(2).return_fields("source")
        )
        .docs
    )


def find_docs_in_redis_not_in_chroma(
    project_id: str = "langchain", delete_docs: bool = False
) -> List[Document]:
    """
    Find documents that are in Redis but not in ChromaDB.

    Args:
        project_id: The project ID to query for
        delete_docs: If True, delete the documents from Redis

    Returns:
        List of document records that are in Redis but not in ChromaDB.
    """
    db = get_vector_store(project_id)
    initialize_project_indices(r, project_id)
    # Wait for the index to be created
    info = r.ft(f"{source_index_name}_{project_id}").info()
    while info["indexing"] == "1":  # type: ignore
        logger.info("Waiting for source index to be created...")
        time.sleep(1)
        info = r.ft(f"{source_index_name}_{project_id}").info()

    redis_docs = (
        r.ft(f"{source_index_name}_{project_id}")
        .search(
            Query("*")
            .dialect(2)
            .return_fields("source")
            .paging(0, 20000)
            .timeout(10000)
        )
        .docs
    )

    docs_not_in_chroma = []
    for doc in redis_docs:
        source_uri = doc.source
        chroma_docs = db.get(where={"source": {"$in": [source_uri]}}, include=[])
        if len(chroma_docs["ids"]) == 0:
            docs_not_in_chroma.append(doc)

            if delete_docs:
                # Delete document from Redis only (since it's not in ChromaDB)
                redis_key = (
                    f"climate-rag::{project_id}::source:{source_uri}"
                    if project_id != "langchain"
                    else f"climate-rag::source:{source_uri}"
                )
                deleted = r.delete(redis_key) == 1
                if deleted:
                    logger.info(f"Deleted document from Redis: {source_uri}")
                else:
                    logger.warning(
                        f"Failed to delete document from Redis: {source_uri}"
                    )

    return docs_not_in_chroma


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean up problematic documents from the database"
    )
    parser.add_argument(
        "--remove-urls",
        nargs="+",
        default=[],
        help="URL(s) to remove from the database",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=400,
        help="Minimum document length (default: 400 characters)",
    )
    parser.add_argument(
        "--project-id",
        type=str,
        help="The project ID to query for.",
        default="langchain",
    )
    parser.add_argument(
        "--find-missing-in-chroma",
        action="store_true",
        help="Find documents in Redis but not in ChromaDB.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete documents found in Redis but not in ChromaDB (use with --find-missing-in-chroma).",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()
    db = get_vector_store(args.project_id)

    # Remove specified URLs
    if args.remove_urls:
        remove_urls(args.remove_urls, db)

    # Find documents with errors
    error_docs = find_error_documents(args.project_id)
    if error_docs:
        logger.info(f"Found {len(error_docs)} total documents with errors")

    # Find short documents
    short_docs = find_short_documents(args.min_length, args.project_id)
    if short_docs:
        logger.info(
            f"Found {len(short_docs)} documents shorter than {args.min_length} characters"
        )

    # Find documents in redis but not in chromadb
    if args.find_missing_in_chroma:
        missing_docs = find_docs_in_redis_not_in_chroma(args.project_id, args.delete)
        if missing_docs:
            action = "Deleted" if args.delete else "Found"
            logger.info(
                f"{action} {len(missing_docs)} documents in Redis but not in ChromaDB:"
            )
            for doc in missing_docs:
                logger.info(f"  - {doc['source']}")


if __name__ == "__main__":
    main()
