import argparse
import os
import shutil
from tools import add_urls_to_db, split_documents, add_to_chroma, get_vector_store, load_documents


CHROMA_PATH = "chroma"
DATA_PATH = "data"

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
        add_urls_to_db([args.url], db)
    else:
        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(db, chunks)



def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
