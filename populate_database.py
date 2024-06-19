import argparse
import os
import shutil
from tools import add_urls_to_db_firecrawl, split_documents, add_to_chroma, get_vector_store, load_documents, clear_database


CHROMA_PATH = "chroma"

def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--url", type=str, help="URL to add to the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    db = get_vector_store()
    if args.url:
        add_urls_to_db_firecrawl([args.url], db)
    else:
        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(db, chunks)





if __name__ == "__main__":
    main()
