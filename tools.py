import hashlib
from ulid import ULID
import os
import logging
import tempfile
from typing import Any, Dict, List, Optional, Sequence
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
import msgspec
from get_embedding_function import get_embedding_function
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import datetime

from cache import r
from helpers import clean_urls, modify_document_source_urls, sanitize_url

from pdf_download import download_urls_in_headed_chrome

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
    elif "Error: Page.goto: Timeout 30000ms exceeded." in page_content:
        return "Timeout exceeded."
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


def add_urls_to_db(
    urls: List[str], db: Chroma, use_firecrawl: bool = False
) -> List[Document]:
    """Add a list of URLs to the database.

    Decide which loader to use based on the URL.
    """

    docs = []
    for url in urls:
        ids_existing = r.keys(f"*{url}")
        # Only add url if it is not already in the database
        if len(ids_existing) == 0:
            if url.lower().endswith(".md"):
                # Can directly download markdown without any processing
                docs += add_urls_to_db_html([url], db)
            else:
                if use_firecrawl:
                    docs += add_urls_to_db_firecrawl([url], db)
                else:
                    if "pdf" in url:
                        jina_docs = add_urls_to_db_html(
                            ["https://r.jina.ai/" + url], db
                        )
                        # Check if the URL has been successfully processed
                        if url in list(map(lambda x: x.metadata["source"], jina_docs)):
                            docs += jina_docs
                        else:
                            # Otherwise, download file using headed chrome
                            temp_dir = tempfile.TemporaryDirectory()
                            try:
                                downloaded_urls = download_urls_in_headed_chrome(
                                    urls=[url], download_dir=temp_dir.name
                                )
                                # Then upload the downloaded file to the database
                                if len(downloaded_urls) > 0:
                                    uploaded_docs = upload_documents(
                                        files=[downloaded_urls[0]["local_path"]], db=db
                                    )
                                    # Change the source to the original URL
                                    modify_document_source_urls(
                                        uploaded_docs[0].metadata["source"], url, db, r
                                    )
                                else:
                                    print(
                                        "Failed to download file via Headed Chrome: ", url
                                    )
                                    uploaded_docs = []
                            except Exception as e:
                                # create ulid for error
                                error_hash = str(ULID())
                                # Save error to redis
                                r.hset(
                                    f"climate-rag::error:{error_hash}",
                                    mapping={
                                        "error": str(e),
                                        "url": url,
                                        "date_added": datetime.datetime.now().isoformat(),
                                        "source": "headed_chrome",
                                    },
                                )
                                print(
                                    f"Error downloading file via Headed Chrome: {url}. Error saved to redis with key {error_hash}"
                                )
                                uploaded_docs = []
                            docs += uploaded_docs
                    else:
                        # use local chrome loader instead
                        chrome_docs = add_urls_to_db_chrome([url], db)
                        # Check if the URL has been successfully processed
                        if url in list(
                            map(lambda x: x.metadata["source"], chrome_docs)
                        ):
                            docs += chrome_docs
                        else:
                            # Otherwise, use jina.ai loader
                            docs += add_urls_to_db_html(
                                ["https://r.jina.ai/" + url], db
                            )
        else:
            print("Already in database: ", url)
    return docs


def add_urls_to_db_html(urls: List[str], db) -> List[Document]:
    from langchain_community.document_loaders import AsyncHtmlLoader
    from chromium import user_agents, Rotator

    user_agent = Rotator(user_agents)

    docs = []
    for url in urls:
        default_header_template = {
            "User-Agent": str(user_agent.get()),
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
        ids_existing = r.keys(f"*{url}")
        # Only add url if it is not already in the database
        if len(ids_existing) == 0:
            print("Adding to database: ", url)
            loader = AsyncHtmlLoader([url], header_template=default_header_template)
            webdocs = loader.load()
            assert len(webdocs) == 1, "Only one document should be returned"
            doc = webdocs[0]
            doc.metadata["source"] = url
            doc.metadata["date_added"] = datetime.datetime.now().isoformat()
            if "r.jina.ai" in url:
                doc.metadata["loader"] = "jina"
            else:
                doc.metadata["loader"] = "html"
            page_errors = check_page_content_for_errors(doc.page_content)
            if page_errors:
                print(f"[HtmlLoader] Error loading {url}: {page_errors}")
            else:
                # If using jina.ai, also fetch html content if file is not a pdf
                if ("r.jina.ai" in url) and ("pdf" not in url):
                    default_header_template["X-Return-Format"] = "html"
                    loader = AsyncHtmlLoader(
                        [url], header_template=default_header_template
                    )
                    webdocs = loader.load()
                    doc.metadata["raw_html"] = webdocs[0].page_content
                add_doc_to_redis(r, doc)
                chunks = split_documents([doc])
                add_to_chroma(db, chunks)
                docs += [doc]
        else:
            print("Already in database: ", url)
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
                    doc.metadata["loader"] = "firecrawl"

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


def add_urls_to_db_chrome(urls: List[str], db, headless=True) -> List[Document]:
    from chromium import AsyncChromiumLoader
    from langchain_community.document_transformers import Html2TextTransformer

    # import nest_asyncio

    # nest_asyncio.apply()

    # Filter urls that are already in the database
    filtered_urls = [
        url for url in urls if len(r.keys("climate-rag::source:" + url)) == 0
    ]
    print("Adding to database: ", filtered_urls)
    loader = AsyncChromiumLoader(urls=filtered_urls, headless=headless)
    docs = loader.load()
    # Cache raw html in redis
    for doc in docs:
        doc.metadata["raw_html"] = doc.page_content
    # Transform the documents to markdown
    html2text = Html2TextTransformer(ignore_links=False)
    docs_transformed = html2text.transform_documents(docs)
    for doc in docs_transformed:
        doc.metadata["date_added"] = datetime.datetime.now().isoformat()
        doc.metadata["loader"] = "chrome"
    docs_to_return: List[Document] = []
    for doc in docs_transformed:
        page_errors = check_page_content_for_errors(doc.page_content)
        if page_errors:
            print(f"[Chrome] Error loading {doc.metadata.get('source')}: {page_errors}")
        else:
            # Cache pre-chunked documents in redis
            add_doc_to_redis(r, doc)
            chunks = split_documents([doc])
            add_to_chroma(db, chunks)
            docs_to_return.append(doc)

    return docs_to_return


def format_docs(docs):
    return "\n\n".join(
        """
Title: {title}
Company: {company_name}
Source: {source}
Content:
{content}

---
""".format(
            title=doc.metadata.get("title", ""),
            company_name=doc.metadata.get("company_name", ""),
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


from schemas import PageMetadata, SearchQuery


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


def upload_documents(files: str | List[str], db) -> List[Document]:
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

    docs: List[Document] = []

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
                url="https://tmpfiles.org/api/v1/upload",
                files={"file": open(file, "rb")},
            )
            # Store filename with URL
            dl_url = (
                "https://tmpfiles.org/dl/"
                + response.json()["data"]["url"].replace("https://tmpfiles.org", "")
                + "#"
                + filename
            )
            print("Uploaded to tmpfiles.org at ", dl_url)
        docs += add_urls_to_db_html([sanitize_url(dl_url)], db=db)
    return docs


def upload_file(
    file_name: str, bucket: str, path: str, object_name: Optional[str] = None
):
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


def extract_metadata_from_source_document(source_text) -> PageMetadata:
    from llms import get_chatbot
    from prompts import metadata_extractor_prompt
    from langchain_core.output_parsers import PydanticOutputParser
    import tiktoken

    from langchain.prompts import PromptTemplate, ChatPromptTemplate

    parser = PydanticOutputParser(pydantic_object=PageMetadata)

    llm = get_chatbot("gpt-4o-mini")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", metadata_extractor_prompt),
            ("user", "{raw_text}"),
        ]
    ).partial(response_format=parser.get_format_instructions())
    extract_chain = prompt | llm | parser

    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    total_token_length = len(enc.encode(source_text))

    # The maximum token length for GPT-4o-mini is 128000 tokens
    # Reduce the length of the text to fit within the token limit
    source_text = source_text[: int(120_000 / total_token_length * len(source_text))]

    metadata = extract_chain.invoke({"raw_text": source_text})

    return metadata


def get_source_document_extra_metadata(
    r,
    source_uri,
    metadata_fields: List[Literal["title", "company_name", "publishing_date"]] = [
        "title"
    ],
) -> Dict[str, Any]:
    """Get generated metadata for a source document from redis. If the metadata is not available, generate it first.

    Args:
        source_uri: The source URI of the document.
        metadata_fields: A list of metadata fields to return.

    Returns:
        Dict[str, any]: Returns a dictionary of metadata fields.
    """
    # Check if metadata is available in redis
    dict_to_return = {}
    for field in metadata_fields:
        field_value = r.hget(f"climate-rag::source:{source_uri}", field)
        if field_value:
            dict_to_return[field] = field_value
        else:
            # Generate metadata from source document
            # Try to get the raw_html or page_content from redis
            source_text = r.hget(f"climate-rag::source:{source_uri}", "raw_html")
            if not source_text:
                source_text = r.hget(
                    f"climate-rag::source:{source_uri}", "page_content"
                )
            # Extract metadata from source document using LLM
            page_metadata = extract_metadata_from_source_document(source_text)
            # Save metadata to redis
            page_metadata_map = page_metadata.dict()
            # Convert publishing date to timestamp
            if page_metadata_map["publishing_date"]:
                page_metadata_map["publishing_date"] = int(
                    datetime.datetime(*page_metadata_map["publishing_date"]).timestamp()
                )
            # Convert key_entities to json
            page_metadata_map["key_entities"] = msgspec.json.encode(
                page_metadata_map["key_entities"]
            )
            # Save metadata to redis
            r.hset(f"climate-rag::source:{source_uri}", mapping=page_metadata_map)

            dict_to_return[field] = page_metadata_map[field]

    return dict_to_return
