from cache import r
from tools import (
    add_urls_to_db,
    delete_document_from_db,
    get_vector_store,
    error_messages,
)
from redis.commands.search.query import Query

if __name__ == "__main__":
    db = get_vector_store()
    all_errors = []

    # Search for all error messages in the source documents
    for error_message, response_message in error_messages.items():
        error_message = (
            error_message.replace(":", "?").replace(",", "?").replace(".", "?")
        )
        results = (
            r.ft("idx:source")
            .search(
                Query(f'@page_content: "{error_message}"' """@page_length:[0 10000]""")
                .dialect(2)
                .return_fields("source")
                .paging(0, 10000)
            )
            .docs
        )
        if len(results) > 0:
            print(
                f"Found {len(results)} source documents with error: {response_message}. Adding to deletion queue."
            )
            all_errors.extend(results)

    # Search for all documents with page_length less than 600 characters
    results = (
        r.ft("idx:source")
        .search(Query("@page_length:[0 600]").dialect(2).return_fields("source")
                .paging(0, 10000))
        .docs
    )
    if len(results) > 0:
        print(f"Found {len(results)} source documents with page_length less than 1000. Adding to deletion queue.")
        all_errors.extend(results)
    breakpoint()

    for error in all_errors:
        delete_document_from_db(error.source, db, r)
