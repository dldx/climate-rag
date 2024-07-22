import os
import logging
from typing import List, Optional, Sequence
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import FireCrawlLoader
import shutil
import glob
from typing import Literal
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from get_embedding_function import get_embedding_function
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import datetime

from cache import r
from helpers import clean_urls

import tiktoken
import re

enc = tiktoken.encoding_for_model("gpt-4o")

web_search_tool = TavilySearchResults(k=3)
DATA_PATH = "data"


from dotenv import load_dotenv

load_dotenv()


def check_page_content_for_errors(page_content: str):
    if "SecurityCompromiseError" in page_content:
        return "SecurityCompromiseError"
    elif "InsufficientBalanceError" in page_content:
        return "InsufficientBalanceError"
    elif "AssertionFailureError" in page_content:
        return "AssertionFailureError"
    elif "TimeoutError" in page_content:
        return "TimeoutError"
    elif "Access Denied" in page_content:
        return "Access Denied"
    elif (page_content == "") or (len(page_content) < 600):
        return "Empty or minimal page content"
    elif bool(
        re.search(
            "(404*.not found)|(page not found)|(page cannot be found)|(HTTP404)|(File or directory not found.)|(Page You Requested Was Not Found)|(Error: Page.goto:)|(404 error)|(404 Not Found)|(404 Page Not Found)|(Error 404)|(404 - File or directory not found)|(HTTP Error 404)|(Not Found - 404)|(404 - Not Found)|(404 - Page Not Found)|(Error 404 - Not Found)|(404 - File Not Found)|(HTTP 404 - Not Found)|(404 - Resource Not Found)",
            page_content,
            re.IGNORECASE,
        )
    ):
        return "Page not found in content"
    elif "Verifying you are human." in page_content:
        return "Page requires human verification"
    else:
        return None

def add_urls_to_db(urls: List[str], db: Chroma) -> List[Document]:
    """Add a list of URLs to the database.

    Decide which loader to use based on the URL.
    """

    docs = []
    for url in urls:
        if url.endswith(".md"):
            # Can directly download markdown without any processing
            docs += add_urls_to_db_html([url], db)
        else:
            docs += add_urls_to_db_firecrawl([url], db)
    return docs

def add_urls_to_db_html(urls: List[str], db):
    from langchain_community.document_loaders import AsyncHtmlLoader

    docs = []
    for url in urls:
        default_header_template = {
            "User-Agent": "",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*"
            ";q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.google.com/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        if ("pdf" in url) and ("r.jina.ai" not in url):
            url = "https://r.jina.ai/" + url
        if "r.jina.ai" in url:
            default_header_template["Authorization"] = (
                f"Bearer {os.environ['JINA_API_KEY']}"
            )
            default_header_template["X-With-Generated-Alt"] = "true"
        # Only add url if it is not already in the database
        if len(r.keys(url)) == 0:
            print("Adding to database: ", url)
            loader = AsyncHtmlLoader([url], header_template=default_header_template)
            webdocs = loader.load()
            for doc in webdocs:
                doc.metadata["source"] = url
                doc.metadata["date_added"] = datetime.datetime.now().isoformat()
            page_errors = check_page_content_for_errors(webdocs[0].page_content)
            if page_errors:
                print(f"[HtmlLoader] Error loading {url}: {page_errors}")
            else:
                for doc in webdocs:
                    add_doc_to_redis(r, doc)
                chunks = split_documents(webdocs)
                add_to_chroma(db, chunks)
                docs += webdocs
    return docs


def add_urls_to_db_firecrawl(urls: List[str], db):
    from langchain_community.vectorstores.utils import filter_complex_metadata

    docs = []
    for url in urls:
        ids_existing = r.keys(f"*{url}")
        # Only add url if it is not already in the database
        if len(ids_existing) == 0:
            print("Adding to database: ", url)
            try:
                loader = FireCrawlLoader(
                    api_key=os.environ["FIRECRAWL_API_KEY"], url=url, mode="scrape"
                )
                webdocs = loader.load()
                for doc in webdocs:
                    doc.metadata["source"] = url
                    doc.metadata["date_added"] = datetime.datetime.now().isoformat()

                page_errors = check_page_content_for_errors(webdocs[0].page_content)
                if page_errors:
                    print(f"[Firecrawl] Error loading {url}: {page_errors}")
                else:
                    webdocs = filter_complex_metadata(webdocs)
                    for doc in webdocs:
                        add_doc_to_redis(r, doc)
                    chunks = split_documents(webdocs)
                    add_to_chroma(db, chunks)
                    docs += webdocs
            except Exception as e:
                print(f"[Firecrawl] Error loading {url}: {e}")
                if (("429" in str(e)) or ("402" in str(e))) and "pdf" not in url:
                    # use local chrome loader instead
                    docs += add_urls_to_db_chrome([url], db)
                elif "502" in str(e):
                    docs += add_urls_to_db_html(["https://r.jina.ai/" + url], db)
                else:
                    docs += add_urls_to_db_html(["https://r.jina.ai/" + url], db)
        else:
            print("Already in database: ", url)

    return docs


def add_doc_to_redis(r, doc):
    doc_dict = {
        **doc.metadata,
        **{
            "date_added": int(
                datetime.datetime.timestamp(datetime.datetime.now(datetime.UTC))
            ),
            "page_content": doc.page_content,
            "page_length": len(doc.page_content),
        },
    }
    r.hset("climate-rag::source:" + doc_dict["source"], mapping=doc_dict)


def add_urls_to_db_chrome(urls: List[str], db):
    from chromium import AsyncChromiumLoader
    from langchain_community.document_transformers import Html2TextTransformer
    import nest_asyncio

    nest_asyncio.apply()

    # Filter urls that are already in the database
    filtered_urls = [url for url in urls if len(r.keys(url)) == 0]
    print("Adding to database: ", filtered_urls)
    loader = AsyncChromiumLoader(urls=filtered_urls)
    docs = loader.load()
    # Cache raw html in redis
    for doc in docs:
        r.hset(
            "climate-rag::source:" + doc.metadata["source"],
            mapping={"raw_html": doc.page_content},
        )
    # Transform the documents to markdown
    html2text = Html2TextTransformer(ignore_links=False)
    docs_transformed = html2text.transform_documents(docs)
    for doc in docs_transformed:
        doc.metadata["date_added"] = datetime.datetime.now().isoformat()
    docs_to_return = []
    for doc in docs_transformed:
        page_errors = check_page_content_for_errors(doc.page_content)
        if page_errors:
            print(f"[Chrome] Error loading {doc.metadata.get('source')}: {page_errors}")
        else:
            # Cache pre-chunked documents in redis
            add_doc_to_redis(r, doc)
            chunks = split_documents([doc])
            add_to_chroma(db, chunks)
            docs_to_return += doc

    return docs_to_return


def format_docs(docs):
    return "\n\n".join(
        "Source: {source}\nContent: {content}\n\n---".format(
            content=doc.page_content,
            source=(
                clean_urls([doc.metadata["source"]], os.environ.get("STATIC_PATH", ""))[
                    0
                ]
                if "source" in doc.metadata.keys()
                else ""
            ),
        )
        for doc in docs
    )


def get_vector_store():
    import chromadb

    client = chromadb.HttpClient(
        host=os.environ["CHROMADB_HOSTNAME"], port=int(os.environ["CHROMADB_PORT"])
    )
    embedding_function = get_embedding_function()

    db = Chroma(
        client=client,
        collection_name="langchain",
        embedding_function=embedding_function,
    )
    return db


def load_documents():
    # Load PDF documents.
    data = []
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    data += document_loader.load()
    # Load Markdown documents.
    md_paths = glob.glob(DATA_PATH + "/**/*.md", recursive=True)
    for md_path in md_paths:
        document_loader = UnstructuredMarkdownLoader(md_path)
        data += document_loader.load()

    return data


def split_documents(
    documents: list[Document],
    splitter: Literal["character", "semantic"] = "semantic",
    max_token_length: int = 3000,
    iter_no: int = 0,
) -> list[Document]:
    import tiktoken

    if splitter == "semantic":
        text_splitter = SemanticChunker(
            get_embedding_function(), breakpoint_threshold_type="percentile"
        )
    elif splitter == "character":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=160,
            length_function=len,
            is_separator_regex=False,
        )
    split_docs = text_splitter.split_documents(documents)

    # Check if any of the docs are too long
    if iter_no < 2:
        for doc in split_docs:
            # Check if token length is too long
            enc = tiktoken.encoding_for_model("gpt-4o")
            if len(enc.encode(doc.page_content)) > max_token_length:
                doc_index = split_docs.index(doc)
                split_docs.remove(doc)
                split_docs.insert(
                    doc_index,
                    split_documents(
                        [doc],
                        splitter=splitter,
                        max_token_length=max_token_length,
                        iter_no=iter_no + 1,
                    ),
                )

    split_docs = [
        item
        for sublist in split_docs
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]
    return split_docs


def add_to_chroma(db: Chroma, chunks: list[Document]):
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


from schemas import SearchQuery


def _unique_documents(documents: Sequence[Document]) -> List[Document]:
    return [doc for i, doc in enumerate(documents) if doc not in documents[:i]]


def retrieve_multiple_queries(queries: List[str], retriever, k: int = -1):
    """Run all LLM generated queries on retriever.

    Args:
        queries: query list

    Returns:
        List of retrieved Documents
    """
    documents = []
    for query in queries:
        docs = retriever.invoke(query)
        documents.extend(docs)

    unique_documents = _unique_documents(documents)[:k]

    return unique_documents


def upload_documents(files: str | List[str], db) -> List[str]:
    """
    Add a document to the database from a local path.

    Args:
        file (str): The path to the file to upload.

    Returns:
        str: The filename of the uploaded file.
    """
    import requests
    import shutil

    UPLOAD_FILE_PATH = os.environ.get("UPLOAD_FILE_PATH", "")
    STATIC_PATH = os.environ.get("STATIC_PATH", "")
    USE_S3 = os.environ.get("USE_S3", False) == "True"

    if type(files) == str:
        files = [files]

    filenames = []

    for file in files:
        filename = file.split("/")[-1]
        # if UPLOAD_FILE_PATH is specified, use it to save the uploaded file
        if UPLOAD_FILE_PATH != "":
            local_path = os.path.join(UPLOAD_FILE_PATH, filename)
            os.makedirs(UPLOAD_FILE_PATH, exist_ok=True)
            # Check if file already exists
            if os.path.exists(local_path):
                return [f"{filename} already exists! Rename the file and try again."]
            shutil.copyfile(file, local_path)

        # if USE_S3 is True, upload the file to S3 (or equivalent like Cloudflare R2)
        if USE_S3 and STATIC_PATH != "":
            bucket = os.environ.get("S3_BUCKET", "")
            object_name = filename
            # Get checksum of file
            import hashlib
            with open(file, "rb") as f:
                # Produce a short hash of the file
                digest = hashlib.file_digest(f, "shake_256").hexdigest(5)
            path = f"docs/{digest}/"
            if not upload_file(file, bucket, path, object_name):
                return [f"Failed to upload {filename} to S3 bucket {bucket}."]
            dl_url = f"{STATIC_PATH}{path}{object_name}"
        else:
            # Otherwise upload the file to tmpfiles.org to load into firecrawl or jina
            response = requests.post(
                url="https://tmpfiles.org/api/v1/upload", files={"file": open(file, "rb")}
            )
            # Store filename with URL
            dl_url = (
                "https://tmpfiles.org/dl/"
                + response.json()["data"]["url"].replace("https://tmpfiles.org", "")
                + "#"
                + filename
            )
            print("Uploaded to tmpfiles.org at ", dl_url)
        add_urls_to_db([dl_url], db)
        filenames.append(filename)
    return filenames


def upload_file(file_name: str, bucket: str, path: str, object_name: Optional[str] = None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    import boto3
    from botocore.exceptions import ClientError

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL", ""),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("S3_ACCESS_KEY_SECRET", ""),
    )
    try:
        response = s3_client.upload_file(file_name, bucket, path + object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True
