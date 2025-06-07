#!/usr/bin/env python3
"""
Database cleanup utility for removing problematic documents and short content.
"""

import argparse
import logging
from typing import List

from redis.commands.search.query import Query

from cache import r, source_index_name
from tools import (
    delete_document_from_db,
    error_messages,
    get_vector_store,
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


def find_error_documents(project_id: str = "langchain") -> List[dict]:
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
) -> List[dict]:
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


if __name__ == "__main__":
    main()
