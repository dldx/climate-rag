import argparse
import os
import shutil
from tools import (
    add_urls_to_db,
    split_documents,
    add_to_chroma,
    get_vector_store,
    load_documents,
    upload_documents,
)


def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument(
        "--urls", nargs="+", default=[], help="URL(s) to add to the database."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=[],
        help="Add file(s) to the database. Files will be uploaded to temporary hosting and then added to the database.",
    )
    parser.add_argument(
        "--force-gemini",
        action="store_true",
        help="Use Gemini to process URLs before adding them to the database. This is better for complex PDFs with many tables.",
    )
    parser.add_argument(
        "--add-table-context",
        action="store_true",
        help="Add additional context to all tables in the document to aid with retrieval.",
    )
    parser.add_argument(
        "--document-prefix",
        type=str,
        default="",
        help="Prefix to add to the documents when uploading to the database.",
    )


    args = parser.parse_args()

    # Create (or update) the data store.
    db = get_vector_store()
    if args.urls:
        for url in args.urls:
            add_urls_to_db([url], db, use_gemini=args.force_gemini, table_augmenter=args.add_table_context, document_prefix=args.document_prefix)
    elif args.files:
        upload_documents(args.files, db, use_gemini=args.force_gemini, table_augmenter=args.add_table_context, document_prefix=args.document_prefix)
    else:
        # For local documents
        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(db, chunks)


if __name__ == "__main__":
    main()
