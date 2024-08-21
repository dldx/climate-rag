from cache import r
from tools import add_urls_to_db, get_vector_store

if __name__ == "__main__":
    db = get_vector_store()
    urls_to_download = []
    for error in r.keys("climate-rag::error:*"):
        if r.hget(error, "status") not in ["in_database", "ignored"]:
            url = r.hget(error, "url")
            if "https://r.jina.ai/" in url:
                r.hset(error, "status", "ignored")
            elif len(r.keys(f"climate-rag::source:*{url}")) > 0:
                r.hset(error, "status", "in_database")
            else:
                urls_to_download.append(url)

    urls_to_download = list(set(urls_to_download))
    print(f"Downloading {len(urls_to_download)} urls:", urls_to_download)
    docs = add_urls_to_db(urls=urls_to_download, db=db)
