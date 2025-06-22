import asyncio
import datetime
import glob
import logging
import os
import re
import shutil
import tempfile
import traceback
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    FireCrawlLoader,
    PyPDFDirectoryLoader,
    UnstructuredMarkdownLoader,
    YoutubeLoader,
)
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from redis import ResponseError
from redis.commands.search.field import NumericField, TagField, TextField

try:
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
except ImportError:
    from redis.commands.search.index_definition import IndexDefinition, IndexType

from ulid import ULID

from climate_rag.cache import (
    ja_source_index_name,
    r,
    source_index_name,
    zh_source_index_name,
)
from climate_rag.get_embedding_function import get_embedding_function
from climate_rag.helpers import (
    clean_up_metadata_object,
    clean_urls,
    modify_document_source_urls,
    sanitize_url,
    upload_file,
)
from climate_rag.pdf_download import (
    download_urls_in_headed_chrome,
    download_urls_with_requests,
)
from climate_rag.schemas import SourceMetadata
from climate_rag.text_splitters import (
    TablePreservingSemanticChunker,
    TablePreservingTextSplitter,
)

logger = logging.getLogger(__name__)
enc = tiktoken.encoding_for_model("gpt-4o")

web_search_tool = TavilySearch(k=3)
DATA_PATH = "data"


load_dotenv()

error_messages = {
    "SecurityCompromiseError": "SecurityCompromiseError",
    "InsufficientBalanceError": "InsufficientBalanceError",
    "AssertionFailureError": "AssertionFailureError",
    "TimeoutError": "TimeoutError",
    "Error: Page.goto: Timeout 30000ms exceeded.": "Timeout exceeded.",
    "Verifying you are human.": "Page requires human verification",
    "please click the box below to let us know you're not a robot": "Page requires human verification",
    "The connection to the origin web server was made, but the origin web server timed out before responding. The likely cause is an overloaded background task, database or application, stressing the resources on your web server.": "Cloudflare timeout error",
}


def store_error_in_redis(url: str, error: str, source: str):
    """
    Store an error in Redis for later analysis.

    Args:
        url (str): The URL that caused the error.
        error (str): The error message.
        source (str): The source function/tool of the error.
    """
    # create ulid for error
    error_hash = str(ULID())
    # Save error to redis
    r.hset(
        f"climate-rag::error:{error_hash}",
        mapping={
            "error": error,
            "url": url,
            "date_added": datetime.datetime.now().isoformat(),
            "source": source,
        },
    )
    print(f"['climate-rag::error:{error_hash}'] Error loading {url}: {error}  ")

    return error_hash


def check_page_content_for_errors(page_content: str):
    for error, return_message in error_messages.items():
        if error in page_content:
            return return_message

    if (page_content == "") or (len(page_content) < 400):
        return "Empty or minimal page content"
    if bool(
        re.search(
            "(404*.not found)|(page not found)|(page cannot be found)|(HTTP404)|(File or directory not found.)|(Page You Requested Was Not Found)|(Error: Page.goto:)|(404 error)|(404 Not Found)|(404 Page Not Found)|(Error 404)|(404 - File or directory not found)|(HTTP Error 404)|(Not Found - 404)|(404 - Not Found)|(404 - Page Not Found)|(Error 404 - Not Found)|(404 - File Not Found)|(HTTP 404 - Not Found)|(404 - Resource Not Found)",
            page_content,
            re.IGNORECASE,
        )
    ):
        return "Page not found in content"
    return None


def add_urls_to_db(
    urls: List[str],
    db: Chroma,
    use_firecrawl: bool = False,
    use_gemini: bool = False,
    table_augmenter: Optional[bool | Callable[[str], str]] = True,
    document_prefix: str = "",
    project_id: str = "langchain",
) -> List[Document]:
    """Add a list of URLs to the database. Decide which loader to use based on the URL.

    Args:
        urls: A list of URLs to add to the database.
        db: The Chroma database instance.
        use_firecrawl: Whether to use FireCrawl to load the URLs.
        use_gemini: Whether to use Gemini to process PDFs.
        table_augmenter: A function to add additional context to tables in the document. If True, use the default table augmenter. If False or None, do not use a table augmenter. If a function, use the provided function.
        document_prefix: A prefix to add to the documents when uploading to the database.
        project_id: The project to add the document to. Defaults to "langchain".

    Returns:
        A list of documents that were added to the database.
    """

    if table_augmenter is True:
        table_augmenter = add_additional_table_context
    elif table_augmenter is False:
        table_augmenter = None

    docs = []
    for url in urls:
        # First check if URL already exists in this project
        ids_existing = r.keys(f"climate-rag::{project_id}::source:*{url}")
        # Only add url if it is not already in the database
        if len(ids_existing) == 0:
            if url.lower().endswith(".md"):
                # Can directly download markdown without any processing
                docs += add_urls_to_db_html(
                    [url],
                    db,
                    table_augmenter=table_augmenter,
                    document_prefix=document_prefix,
                    project_id=project_id,
                )
            elif (
                url.lower().endswith(".xls")
                or url.lower().endswith(".xlsx")
                or url.lower().endswith(".zip")
            ):
                # Cannot load excel files right now
                store_error_in_redis(url, "Cannot load excel files", "add_urls_to_db")
            # Check if it is a youtube url
            elif "youtube.com/watch?v=" in url:
                # Use the youtube loader
                docs += add_urls_to_db_youtube([url], db, project_id=project_id)
            else:
                if use_firecrawl:
                    docs += add_urls_to_db_firecrawl(
                        [url],
                        db,
                        table_augmenter=table_augmenter,
                        document_prefix=document_prefix,
                        project_id=project_id,
                    )
                elif use_gemini:
                    # Use Gemini to process the PDF
                    # download file using headed chrome
                    temp_dir = tempfile.TemporaryDirectory()
                    try:
                        # Download the file using headed chrome if file is not an image
                        if ".png" not in url and ".jpg" not in url:
                            downloaded_urls = download_urls_in_headed_chrome(
                                urls=[url], download_dir=temp_dir.name
                            )
                        else:
                            downloaded_urls = download_urls_with_requests(
                                urls=[url], download_dir=temp_dir.name
                            )
                        # Try using Gemini to process the PDF
                        gemini_docs = add_document_to_db_via_gemini(
                            downloaded_urls[0]["local_path"],
                            url,
                            db,
                            table_augmenter=table_augmenter,
                            document_prefix=document_prefix,
                            project_id=project_id,
                        )
                        docs += gemini_docs

                    except Exception:
                        store_error_in_redis(
                            url, str(traceback.format_exc()), "headed_chrome"
                        )
                        downloaded_urls = []

                else:
                    if "pdf" in url:
                        jina_docs = add_urls_to_db_html(
                            ["https://r.jina.ai/" + url],
                            db,
                            table_augmenter=table_augmenter,
                            document_prefix=document_prefix,
                            project_id=project_id,
                        )
                        # Check if the URL has been successfully processed
                        if url in list(
                            map(
                                lambda x: x.metadata["source"].replace(
                                    "https://r.jina.ai/", ""
                                ),
                                jina_docs,
                            )
                        ):
                            docs += jina_docs
                        else:
                            # If file is stored on S3 server, we probably need to use Gemini to process it since jina.ai likely failed
                            if os.environ["STATIC_PATH"] in url:
                                # Try using Gemini to process the PDF
                                uploaded_docs = add_document_to_db_via_gemini(
                                    url,
                                    url,
                                    db,
                                    table_augmenter=table_augmenter,
                                    document_prefix=document_prefix,
                                    project_id=project_id,
                                )
                            else:
                                # Otherwise, download file using headed chrome
                                temp_dir = tempfile.TemporaryDirectory()
                                try:
                                    downloaded_urls = download_urls_in_headed_chrome(
                                        urls=[url], download_dir=temp_dir.name
                                    )
                                    # Then upload the downloaded file to the database
                                    if len(downloaded_urls) > 0:
                                        # Upload documents to server and then process it
                                        uploaded_docs = upload_documents(
                                            files=[downloaded_urls[0]["local_path"]],
                                            db=db,
                                            use_gemini=use_gemini,
                                            table_augmenter=table_augmenter,
                                            document_prefix=document_prefix,
                                            project_id=project_id,
                                        )
                                        if len(uploaded_docs) > 0:
                                            # Change the source to the original URL
                                            modify_document_source_urls(
                                                uploaded_docs[0].metadata["source"],
                                                url,
                                                db,
                                                r,
                                                project_id=project_id,
                                            )
                                        else:
                                            # Try using Gemini to process the PDF
                                            uploaded_docs = (
                                                add_document_to_db_via_gemini(
                                                    downloaded_urls[0]["local_path"],
                                                    url,
                                                    db,
                                                    table_augmenter=table_augmenter,
                                                    document_prefix=document_prefix,
                                                    project_id=project_id,
                                                )
                                            )
                                    else:
                                        raise Exception(
                                            f"Failed to download file via Headed Chrome: {url}"
                                        )
                                except Exception:
                                    store_error_in_redis(
                                        url,
                                        str(traceback.format_exc()),
                                        "headed_chrome",
                                    )
                                    uploaded_docs = []
                            docs += uploaded_docs
                    else:
                        # use local chrome loader instead
                        chrome_docs = asyncio.run(
                            add_urls_to_db_chrome(
                                [url],
                                db,
                                table_augmenter=table_augmenter,
                                document_prefix=document_prefix,
                                project_id=project_id,
                            )
                        )
                        # Check if the URL has been successfully processed
                        if url in list(
                            map(lambda x: x.metadata["source"], chrome_docs)
                        ):
                            docs += chrome_docs
                        else:
                            # Otherwise, use jina.ai loader
                            docs += add_urls_to_db_html(
                                ["https://r.jina.ai/" + url],
                                db,
                                table_augmenter=table_augmenter,
                                document_prefix=document_prefix,
                                project_id=project_id,
                            )

        else:
            print(f"Already in database (project {project_id}): {url}")
    # Fetch additional metadata
    # There should only be one url in this but just in case
    for i in set(map(lambda x: x.metadata["source"], docs)):
        print(f"Fetching additional metadata for: {i} (project {project_id})")
        get_source_document_extra_metadata(r, i, project_id=project_id)
    return docs


def add_urls_to_db_html(
    urls: List[str], db, table_augmenter, document_prefix, project_id
) -> List[Document]:
    from langchain_community.document_loaders import AsyncHtmlLoader

    from climate_rag.chromium import Rotator, user_agents

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
                doc = clean_document_contents([doc])[0]
                # Add prefix to document
                if len(document_prefix) > 0:
                    doc.page_content = document_prefix + "\n" + doc.page_content
                chunks = split_documents([doc], table_augmenter=table_augmenter)
                add_to_chroma(db, chunks)
                add_doc_to_redis(r, doc, project_id=project_id)
                docs += [doc]
        else:
            print("Already in database: ", url)
    return docs


def add_urls_to_db_youtube(urls: List[str], db, project_id: str) -> List[Document]:
    docs = []
    for url in urls:
        ids_existing = r.keys(f"*{url}")
        # Only add url if it is not already in the database
        if len(ids_existing) == 0:
            print("Adding to database: ", url)

            loader = YoutubeLoader.from_youtube_url(
                youtube_url=url,
                # add_video_info=True,
                language=["en"],
                translation="en",
            )
            doc = loader.load()[0]
            doc.metadata["source"] = url
            doc.metadata["date_added"] = datetime.datetime.now().isoformat()
            doc.metadata["loader"] = "youtube"
            page_errors = check_page_content_for_errors(doc.page_content)
            if page_errors:
                print(f"[Youtube] Error loading {url}: {page_errors}")
            else:
                chunks = split_documents([doc])
                add_to_chroma(db, chunks)
                add_doc_to_redis(r, doc, project_id=project_id)
                docs += [doc]
        else:
            print("Already in database: ", url)
    return docs


def add_document_to_db_via_gemini(
    doc_uri: os.PathLike | str,
    original_uri: str,
    db,
    table_augmenter: Optional[Callable[[str], str]] = None,
    document_prefix: str = "",
    project_id: str = "langchain",
) -> List[Document]:
    """
    Add a document to the database using Google Gemini. Gemini is used to process complex PDFs with many tables.

    Args:
        doc_uri: The URI of the document to add.
        original_uri: The original URI of the document.
        db: The Chroma database instance.
        table_augmenter: A function to add additional context to tables in the document. If True, use the default table augmenter. If False or None, do not use a table augmenter. If a function, use the provided function.
        document_prefix: A prefix to add to the documents when uploading to the database. Useful for adding context that is not in the document itself.
        project_id: The project to add the document to. Defaults to "langchain".

    Returns:
        A list of documents that were added to the database.
    """
    from climate_rag.process_pdf_via_gemini import process_pdf_via_gemini

    docs = []
    ids_existing = r.keys(f"*{original_uri}")
    # Only add url if it is not already in the database
    if len(ids_existing) == 0:
        print("[Gemini] Adding to database: ", original_uri)
        try:
            pdf_metadata, pdf_contents = process_pdf_via_gemini(
                doc_uri, document_prefix=document_prefix
            )
            # Prefix the document with additional context
            if len(document_prefix) > 0:
                # If prefix has not already been added, add it now
                if not pdf_contents.startswith(document_prefix):
                    pdf_contents = document_prefix + "\n" + pdf_contents
            doc = Document(
                metadata={
                    "source": original_uri,
                    "date_added": datetime.datetime.now().isoformat(),
                    "loader": "gemini",
                },
                page_content=pdf_contents,
            )
            page_errors = check_page_content_for_errors(doc.page_content)
            if page_errors:
                store_error_in_redis(original_uri, page_errors, "gemini")
            else:
                doc = clean_document_contents([doc])[0]
                chunks = split_documents(
                    filter_complex_metadata([doc]), table_augmenter=table_augmenter
                )
                add_to_chroma(db, chunks)
                doc.metadata = {
                    **clean_up_metadata_object(pdf_metadata),
                    **doc.metadata,
                }
                doc.metadata["fetched_additional_metadata"] = "true"
                add_doc_to_redis(r, doc, project_id=project_id)
                docs += [doc]
        except Exception:
            store_error_in_redis(original_uri, str(traceback.format_exc()), "gemini")
    else:
        print("Already in database: ", original_uri)

    return docs


def add_urls_to_db_firecrawl(
    urls: List[str],
    db,
    table_augmenter=None,
    document_prefix="",
    project_id="langchain",
) -> List[Document]:
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
                    webdocs = clean_document_contents(webdocs)
                    for i, doc in enumerate(webdocs):
                        # Add prefix to document
                        if len(document_prefix) > 0:
                            doc.page_content = document_prefix + "\n" + doc.page_content
                            webdocs[i] = doc
                        # Then cache the document in redis
                        add_doc_to_redis(r, doc, project_id=project_id)
                    chunks = split_documents(webdocs, table_augmenter=table_augmenter)
                    add_to_chroma(db, chunks)
                    docs += webdocs
            except Exception as e:
                print(f"[Firecrawl] Error loading {url}: {e}")
                if (("429" in str(e)) or ("402" in str(e))) and "pdf" not in url:
                    # use local chrome loader instead
                    docs += add_urls_to_db(
                        [url],
                        db,
                        table_augmenter=table_augmenter,
                        document_prefix=document_prefix,
                        project_id=project_id,
                    )
                elif "502" in str(e):
                    docs += add_urls_to_db_html(
                        ["https://r.jina.ai/" + url],
                        db,
                        table_augmenter=table_augmenter,
                        document_prefix=document_prefix,
                        project_id=project_id,
                    )
                else:
                    docs += add_urls_to_db_html(
                        ["https://r.jina.ai/" + url],
                        db,
                        table_augmenter=table_augmenter,
                        document_prefix=document_prefix,
                        project_id=project_id,
                    )
        else:
            print("Already in database: ", url)

    return docs


def add_doc_to_redis(r, doc, project_id):
    """
    Add a document to redis with relevant keys.

    Args:
        r: The redis connection.
        doc: The document to add.
        project_id: The project ID to add the document to.
    """
    doc_dict = {
        **doc.metadata,
        **{
            "date_added": int(
                datetime.datetime.timestamp(datetime.datetime.now(datetime.UTC))
            ),
            "page_content": doc.page_content,
            "page_length": len(doc.page_content),
            "project_id": project_id,
        },
    }
    r.hset(f"climate-rag::{project_id}::source:" + doc_dict["source"], mapping=doc_dict)


async def add_urls_to_db_chrome(
    urls: List[str],
    db,
    headless=True,
    table_augmenter=None,
    document_prefix="",
    project_id="langchain",
) -> List[Document]:
    import asyncio

    from langchain_community.document_loaders import (
        AsyncChromiumLoader,
        AsyncHtmlLoader,
    )
    from langchain_community.document_transformers import Html2TextTransformer

    from climate_rag.chromium import Rotator, user_agents

    user_agent = Rotator(user_agents)
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

    # Filter urls that are already in the database
    filtered_urls = [
        url for url in urls if len(r.keys("climate-rag::source:" + url)) == 0
    ]
    print("Adding to database: ", filtered_urls)
    # Load document with
    html_loader = AsyncHtmlLoader(
        filtered_urls, header_template=default_header_template
    )
    docs_html = html_loader.aload()
    original_chromium_loader = AsyncChromiumLoader(
        urls=filtered_urls, headless=headless, user_agent=str(user_agent.get())
    )
    docs_original_chromium = original_chromium_loader.aload()

    docs = await asyncio.gather(docs_html, docs_original_chromium)
    docs = list(
        map(
            lambda x: x[0] if len(x[0].page_content) > len(x[1].page_content) else x[1],
            zip(*docs),
        )
    )

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
            doc = clean_document_contents([doc])[0]
            # Add prefix to document
            if len(document_prefix) > 0:
                doc.page_content = document_prefix + "\n" + doc.page_content
            chunks = split_documents([doc], table_augmenter=table_augmenter)
            add_to_chroma(db, chunks)
            # Cache pre-chunked documents in redis
            add_doc_to_redis(r, doc, project_id=project_id)
            docs_to_return.append(doc)

    return docs_to_return


def delete_document_from_db(source_uri: str, db, r) -> bool:
    """
    Delete a document from the database.

    Args:
        source_uri: The URI of the document to delete.
        db: The Chroma database instance.
        r: The Redis connection.

    Returns:
        bool: True if the document was deleted, False if it was not found.
    """
    from redis.commands.search.query import Query

    # Find original key using redis FT.SEARCH
    ids = r.ft(source_index_name).search(
        Query(f'@source:"{source_uri}"').dialect(2).return_fields("id")
    )
    if len(ids.docs) == 0:
        print(f"Document not found in redis: {source_uri}")
        return False
    selected_doc = 0
    if len(ids.docs) > 1:
        print(f"Multiple documents found in redis: {source_uri}")
        # Get user input to select the correct document
        for i, doc in enumerate(ids.docs):
            print(f"{i}: {doc.id}")
        selected_doc = input("Select the document to delete: ")
        selected_doc = int(selected_doc)
        if selected_doc < 0 or selected_doc >= len(ids.docs):
            print("Invalid selection")
            return False
    actual_source_uri = ids.docs[selected_doc].id.replace("climate-rag::source:", "")

    # Delete document from redis
    existed = r.delete(f"climate-rag::source:{actual_source_uri}") == 1
    if existed:
        print(f"Deleted document from redis: {actual_source_uri}")
    else:
        print(f"Document not found in redis: {actual_source_uri}")
    # Delete document from chroma db
    docs = db.get(where={"source": {"$in": [actual_source_uri]}}, include=[])
    if len(docs["ids"]) > 0:
        db.delete(ids=docs["ids"])
        print(
            f"""Deleted {len(docs["ids"])} documents from chroma db: {actual_source_uri}"""
        )
    else:
        print(f"Document not found in chroma db: {actual_source_uri}")

    return existed


def get_sources_based_on_filter(
    rag_filter: str,
    r: Any,
    limit: int = 10_000,
    page_no: int = 0,
    project_id: str = "langchain",
) -> List[str]:
    """
    Get all sources from redis based on a filter.

    Args:
        rag_filter: The filter to use.
        r: The redis connection.
        limit (optional): Number of results to get on one page
        page_no (optional): Page of results to get
        project_id (optional): The project ID to use. Defaults to "langchain"

    Returns:
        List[str]: A list of source URIs.
    """
    import time

    initialize_project_indices(r, project_id)
    # Wait for the index to be created
    info = r.ft(f"{source_index_name}_{project_id}").info()
    while info["indexing"] == "1":  # type: ignore
        logger.info("Waiting for source index to be created...")
        time.sleep(1)
        info = r.ft(f"{source_index_name}_{project_id}").info()

    from redis.commands.search.query import Query

    if bool(re.search(r"@.+:.+", rag_filter)) is False:
        rag_filter = f'@source:"{rag_filter}"'
    # Replace stopwords with empty string because redis doesn't find urls with stopwords :(
    stopwords = [
        "a",
        "is",
        "the",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    ]
    # If the rag filter is a url, remove stopwords and protocol
    if bool(re.search(r"https?://", rag_filter)):
        rag_filter = re.sub(r"\b(?:{})\b".format("|".join(stopwords)), "", rag_filter)
        rag_filter = re.sub(r"https?://", "", rag_filter)

    # Return early if the filter is empty
    if rag_filter == "@source:()":
        return []

    # Get project-specific index names
    project_source_index_name = f"{source_index_name}_{project_id}"
    project_zh_source_index_name = f"{zh_source_index_name}_{project_id}"
    project_ja_source_index_name = f"{ja_source_index_name}_{project_id}"

    # Check if project-specific indices exist, if not, create them
    try:
        r.ft(project_source_index_name).info()
        use_project_index = True
    except ResponseError:
        print(f"Project {project_id} does not have a project-specific index")
        use_project_index = False
    print(
        f"Getting sources based on filter: {rag_filter} for project: {project_id} with project index: {project_source_index_name}"
    )

    # Get all sources from redis based on FT.SEARCH
    try:
        if use_project_index:
            index_prefix = (
                f"climate-rag::{project_id}::source:"
                if project_id != "langchain"
                else "climate-rag::source:"
            )
            source_list = (
                [
                    doc.id.replace(index_prefix, "")
                    for doc in r.ft(project_source_index_name)
                    .search(
                        Query(rag_filter)
                        .dialect(2)
                        .paging(
                            max(0, (page_no - 1)) * limit,
                            limit + max(0, (page_no - 1)) * limit,
                        )
                        .timeout(5000)
                    )
                    .docs
                ]
                + [
                    doc.id.replace(index_prefix, "")
                    for doc in r.ft(project_zh_source_index_name)
                    .search(
                        Query(rag_filter)
                        .dialect(2)
                        .paging(
                            max(0, (page_no - 1)) * limit,
                            limit + max(0, (page_no - 1)) * limit,
                        )
                        .timeout(5000)
                    )
                    .docs
                ]
                + [
                    doc.id.replace(index_prefix, "")
                    for doc in r.ft(project_ja_source_index_name)
                    .search(
                        Query(rag_filter)
                        .dialect(2)
                        .paging(
                            max(0, (page_no - 1)) * limit,
                            limit + max(0, (page_no - 1)) * limit,
                        )
                        .timeout(5000)
                    )
                    .docs
                ]
            )
        else:
            # Fall back to default indices
            index_prefix = "climate-rag::source:"
            source_list = (
                [
                    doc.id.replace(index_prefix, "")
                    for doc in r.ft(source_index_name)
                    .search(
                        Query(rag_filter)
                        .dialect(2)
                        .paging(
                            max(0, (page_no - 1)) * limit,
                            limit + max(0, (page_no - 1)) * limit,
                        )
                        .timeout(5000)
                    )
                    .docs
                ]
                + [
                    doc.id.replace(index_prefix, "")
                    for doc in r.ft(zh_source_index_name)
                    .search(
                        Query(rag_filter)
                        .dialect(2)
                        .paging(
                            max(0, (page_no - 1)) * limit,
                            limit + max(0, (page_no - 1)) * limit,
                        )
                        .timeout(5000)
                    )
                    .docs
                ]
                + [
                    doc.id.replace(index_prefix, "")
                    for doc in r.ft(ja_source_index_name)
                    .search(
                        Query(rag_filter)
                        .dialect(2)
                        .paging(
                            max(0, (page_no - 1)) * limit,
                            limit + max(0, (page_no - 1)) * limit,
                        )
                        .timeout(5000)
                    )
                    .docs
                ]
            )
        source_list = list(set(source_list))
        print(f"Found {len(source_list)} sources: {source_list}")
    except ResponseError:
        print("Redis error:", str(traceback.format_exc()))
        source_list = []
    return source_list


def compile_docs_metadata(docs: list[Document]) -> list[Dict[str, str]]:
    """
    Compile metadata from a list of documents.

    Args:
        docs: A list of documents, each containing metadata. Metadata should contain the following:
            - title
            - company_name
            - publishing_date
            - source

    Returns:
        A list of metadata dictionaries.
    """
    metadata = []
    for doc in docs:
        doc_metadata = dict(
            title=(
                doc.metadata.get("title", "")
                if not pd.isna(doc.metadata.get("title"))
                else ""
            ),
            company_name=(
                doc.metadata.get("company_name", "")
                if not pd.isna(doc.metadata.get("company_name"))
                else ""
            ),
            publishing_date=(
                doc.metadata.get("publishing_date", "")
                if not pd.isna(doc.metadata.get("publishing_date"))
                else ""
            ),
            content=doc.page_content,
            source=(
                clean_urls([doc.metadata["source"]], os.environ.get("STATIC_PATH", ""))[
                    0
                ]
                if "source" in doc.metadata.keys()
                else ""
            ),
        )
        metadata.append(doc_metadata)
    return metadata


def format_docs(docs):
    """
    Format a list of documents into a string.

    Args:
        docs: A list of documents, each containing metadata. Metadata should contain the following:
            - title
            - company_name
            - publishing_date
            - source
            - content

    Returns:
        A formatted string containing the metadata and content of each document, ready to go into an LLM as context.
    """

    formatted_docs = "\n\n".join(
        """
Title: {title}
Company: {company_name}
Source: {source}
Date published: {publishing_date}
Content:
{content}

---
""".format(**doc_metadata)
        for doc_metadata in compile_docs_metadata(docs)
    )

    return formatted_docs


def get_vector_store(
    project_id: str = "langchain", embeddings_model: Optional[str] = None
) -> Chroma:
    """
    Get a vector store for a project.

    Args:
        project_id: The project ID to use. Defaults to "langchain".
        embeddings_model: The model to use for embeddings. See `get_embedding_function` for options. Defaults to None.

    Returns:
        A vector store for the project.
    """
    import chromadb

    client = chromadb.HttpClient(
        host=os.environ["CHROMADB_HOSTNAME"], port=int(os.environ["CHROMADB_PORT"])
    )
    if embeddings_model:
        embedding_function = get_embedding_function(model=embeddings_model)
    else:
        embedding_function = get_embedding_function()

    db = Chroma(
        client=client,
        collection_name=project_id,
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


def clean_document_contents(
    docs: list[Document], remove_table_context=False
) -> list[Document]:
    """
    Clean the contents of a list of documents, removing excessive whitespace and characters that might be artifacts of OCR.
    Also removes table context if specified.

    Args:
        docs: A list of documents to clean.
        remove_table_context: Whether to remove table context.
    Returns:
        A list of cleaned documents.
    """
    MAX_REPEATS_TO_KEEP = 40
    # This pattern matches any character repeated more than MAX_REPEATS_TO_KEEP times
    pattern = rf"((.)\2{{{MAX_REPEATS_TO_KEEP - 1}}})\2{{1,}}"
    table_context_pattern = r"<table_context>.*?</table_context>"

    for doc in docs:
        # Replace excessive characters with just one instance of the character
        doc.page_content = re.sub(pattern, r"\1", doc.page_content)
        # Remove excessive newlines
        doc.page_content = re.sub(r"\n{2,}", "\n", doc.page_content)
        # Remove table context if specified
        if remove_table_context:
            doc.page_content = re.sub(
                table_context_pattern, "", doc.page_content, flags=re.DOTALL
            )
    return docs


def split_documents(
    documents: list[Document],
    splitter: Literal["character", "semantic", "auto"] = "auto",
    max_token_length: int = 3000,
    iter_no: int = 0,
    table_augmenter: Optional[Callable[[str], str]] = None,
) -> list[Document]:
    """
    Split a list of documents into smaller chunks to store in vector database.

    Args:
        documents: A list of documents to split.
        splitter: The text splitter to use. Can be "semantic", "character", or "auto".
                 "auto" automatically selects based on document token length (>100k tokens uses character, otherwise semantic).
                 "semantic" is smarter but slower and more expensive. "character" is faster but dumb.
        max_token_length: The maximum token length for each chunk. If a chunk is longer than this, it will be split further.
        iter_no: The iteration number. Used to prevent infinite recursion.
        table_augmenter: A function to add additional context to tables in the document.

    Returns:
        A list of split documents.
    """
    import mimetypes
    from pathlib import Path

    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-4o")

    documents_to_avoid_splitting = []
    for doc in documents[:]:  # Create a copy to iterate over
        # Check if the filetype is a CSV, TSV, JSON, etc
        if doc.metadata.get("source") and (
            mimetypes.guess_type(Path(doc.metadata["source"]).name)[0]
            in [
                "text/csv",
                "text/tab-separated-values",
                "application/json",
            ]
        ):
            logger.info(
                f"Skipping splitting for {doc.metadata['source']} as it is a {mimetypes.guess_type(Path(doc.metadata['source']).name)[0]} file"
            )
            documents_to_avoid_splitting.append(doc)
            documents.remove(doc)

    # If auto-select mode, group documents by appropriate splitter
    if splitter == "auto":
        TOKEN_THRESHOLD = 60_000  # 100k tokens threshold

        semantic_docs = []
        character_docs = []

        for doc in documents:
            doc_tokens = len(enc.encode(doc.page_content))
            if doc_tokens > TOKEN_THRESHOLD:
                logger.info(
                    f"Document {doc.metadata.get('source', 'unknown')} has {doc_tokens:,} tokens, using character splitter"
                )
                character_docs.append(doc)
            else:
                logger.info(
                    f"Document {doc.metadata.get('source', 'unknown')} has {doc_tokens:,} tokens, using semantic splitter"
                )
                semantic_docs.append(doc)

        split_docs = []

        # Process semantic documents (smaller documents)
        try:
            # Try to use Qwen embeddings if available as it is faster and cheaper
            logger.info("Using Qwen3 4B embeddings for semantic splitting")
            semantic_embeddings_function = get_embedding_function(
                model="Qwen/Qwen3-Embedding-4B"
            )
        except ValueError:
            semantic_embeddings_function = get_embedding_function()
        if semantic_docs:
            text_splitter = TablePreservingSemanticChunker(
                embeddings=semantic_embeddings_function,
                breakpoint_threshold_type="percentile",
                chunk_size=max_token_length,
                length_function=lambda x: len(enc.encode(x)),
                table_augmenter=table_augmenter,
            )
            split_docs.extend(text_splitter.split_documents(semantic_docs))

        # Process character documents (bigger documents)
        if character_docs:
            text_splitter = TablePreservingTextSplitter(
                chunk_size=max_token_length,
                chunk_overlap=0,
                length_function=lambda x: len(enc.encode(x)),
                is_separator_regex=False,
                table_augmenter=None,  # No table augmenter for bigger documents
            )
            split_docs.extend(text_splitter.split_documents(character_docs))

    else:
        # Original behavior for explicit splitter choice
        if splitter == "semantic":
            text_splitter = TablePreservingSemanticChunker(
                embeddings=get_embedding_function(),
                breakpoint_threshold_type="percentile",
                chunk_size=max_token_length,
                length_function=lambda x: len(enc.encode(x)),
                table_augmenter=table_augmenter,
            )
        elif splitter == "character":
            text_splitter = TablePreservingTextSplitter(
                chunk_size=max_token_length,
                chunk_overlap=0,
                length_function=lambda x: len(enc.encode(x)),
                is_separator_regex=False,
                table_augmenter=table_augmenter,
            )
        split_docs = text_splitter.split_documents(documents)

    # Check if any of the docs are too long
    if iter_no < 2:
        docs_to_resplit = []
        for doc in split_docs[:]:  # Create a copy to iterate over
            # Check if token length is too long
            if len(enc.encode(doc.page_content)) > max_token_length:
                docs_to_resplit.append(doc)
                split_docs.remove(doc)

        # Resplit oversized documents
        for doc in docs_to_resplit:
            resplit_docs = split_documents(
                [doc],
                splitter="character",  # Force character splitter for oversized chunks
                max_token_length=max_token_length,
                iter_no=iter_no + 1,
                table_augmenter=table_augmenter,
            )
            split_docs.extend(resplit_docs)

    # Flatten any nested lists (shouldn't happen with the new implementation, but keeping for safety)
    split_docs = [
        item
        for sublist in split_docs
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]
    # Add back the documents that should not be split
    split_docs.extend(documents_to_avoid_splitting)
    return split_docs


def add_to_chroma(db: Chroma, chunks: list[Document]):
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Only add documents that don't exist in the DB.
    document_not_in_db = (
        len(db.get(ids=[chunk.metadata["id"] for chunk in chunks_with_ids])["ids"]) == 0
    )

    if document_not_in_db:
        print(f"👉 Adding new documents: {len(chunks_with_ids)}")

        # Batch chunks to stay under 250,000 tokens per request
        MAX_TOKENS_PER_BATCH = 10_000
        current_batch = []
        current_batch_tokens = 0
        batch_num = 1

        for chunk in chunks_with_ids:
            # Count tokens in the chunk content
            chunk_tokens = len(enc.encode(chunk.page_content))

            # If adding this chunk would exceed the limit, process current batch
            if current_batch and (
                current_batch_tokens + chunk_tokens > MAX_TOKENS_PER_BATCH
            ):
                print(
                    f"📦 Processing batch {batch_num} with {len(current_batch)} chunks ({current_batch_tokens:,} tokens)"
                )
                batch_ids = [chunk.metadata["id"] for chunk in current_batch]
                db.add_documents(current_batch, ids=batch_ids)

                # Start new batch
                current_batch = [chunk]
                current_batch_tokens = chunk_tokens
                batch_num += 1
            else:
                # Add chunk to current batch
                current_batch.append(chunk)
                current_batch_tokens += chunk_tokens

        # Process remaining chunks in the final batch
        if current_batch:
            print(
                f"📦 Processing final batch {batch_num} with {len(current_batch)} chunks ({current_batch_tokens:,} tokens)"
            )
            batch_ids = [chunk.metadata["id"] for chunk in current_batch]
            db.add_documents(current_batch, ids=batch_ids)

        print(f"✅ Successfully added all documents in {batch_num} batch(es)")
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


def _unique_documents(documents: Sequence[Document]) -> List[Document]:
    return [doc for i, doc in enumerate(documents) if doc not in documents[:i]]


async def retrieve_multiple_queries(queries: List[str], retriever, k: int = -1):
    """Run all LLM generated queries on retriever asynchronously.

    Args:
        queries: query list
        retriever: The retriever object (should support ainvoke method)
        k: Maximum number of unique documents to return

    Returns:
        List of retrieved Documents
    """
    # Run all queries concurrently
    tasks = [retriever.ainvoke(query) for query in queries]
    results = await asyncio.gather(*tasks)

    # Flatten the results
    documents = []
    for docs in results:
        documents.extend(docs)

    unique_documents = _unique_documents(documents)[:k]

    return unique_documents


def upload_documents(
    files: str | List[str],
    db,
    use_gemini: bool = False,
    table_augmenter=None,
    document_prefix="",
    project_id="langchain",
) -> List[Document | str]:
    """
    Add a document to the database from a local path.

    Args:
        file (str): The path to the file to upload.
        db: The Chroma database instance.
        use_gemini: Whether to use Gemini to process the PDF.
        table_augmenter: A function to add additional context to tables in the document. If True, use the default table augmenter. If False or None, do not use a table augmenter. If a function, use the provided function.
        document_prefix: A prefix to add to the documents when uploading to the database.
        project_id: The project to add the document to. Defaults to "langchain".

    Returns:
        A list of documents that were added to the database.
    """
    import requests

    UPLOAD_FILE_PATH = os.environ.get("UPLOAD_FILE_PATH", "")
    STATIC_PATH = os.environ.get("STATIC_PATH", "")
    USE_S3 = os.environ.get("USE_S3", False) == "True"

    if type(files) is str:
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
        docs += add_urls_to_db(
            [sanitize_url(dl_url)],
            db=db,
            use_gemini=use_gemini,
            table_augmenter=table_augmenter,
            document_prefix=document_prefix,
            project_id=project_id,
        )
    return docs


def extract_metadata_from_source_document(source_text) -> SourceMetadata:
    import tiktoken
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from climate_rag.llms import get_chatbot, get_max_token_length
    from climate_rag.prompts import metadata_extractor_prompt

    parser = PydanticOutputParser(pydantic_object=SourceMetadata)

    llm = get_chatbot("gemini-2.5-flash")
    max_token_length = get_max_token_length("gemini-2.5-flash")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", metadata_extractor_prompt),
            ("user", "{raw_text}"),
        ]
    ).partial(response_format=parser.get_format_instructions())
    extract_chain = prompt | llm | parser

    enc = tiktoken.encoding_for_model("gpt-4o")
    total_token_length = len(enc.encode(source_text))

    # Reduce the length of the text to fit within the token limit and speed things up
    source_text = source_text[
        : int(max_token_length / total_token_length * len(source_text))
    ]

    metadata = extract_chain.invoke({"raw_text": source_text})

    return metadata


def get_source_document_extra_metadata(
    r,
    source_uri,
    metadata_fields: List[
        Literal["title", "company_name", "publishing_date", "source"]
    ] = ["title"],
    use_llm: bool = True,
    project_id: str = "langchain",
) -> Dict[str, Any]:
    from langchain_core.exceptions import OutputParserException

    """Get generated metadata for a source document from redis. If the metadata is not available, generate it first.

    Args:
        source_uri: The source URI of the document.
        metadata_fields: A list of metadata fields to return.
        use_llm: Whether to use an LLM to generate metadata if it's not available.
        project_id: The project ID to use. Defaults to "langchain".

    Returns:
        Dict[str, any]: Returns a dictionary of metadata fields.
    """
    # Check if metadata is available in redis
    dict_to_return = {}
    redis_key = f"climate-rag::{project_id}::source:{source_uri}"
    # If the project is langchain, and the metadata is not available, use the old key
    if (project_id == "langchain") and (r.exists(redis_key) == 0):
        redis_key = f"climate-rag::source:{source_uri}"
    field_map = dict(
        zip(
            metadata_fields,
            r.hmget(redis_key, *metadata_fields),
        )
    )
    for field_key, field_value in field_map.items():
        if field_value is not None:
            pass
        elif use_llm and (
            r.hget(
                redis_key,
                "fetched_additional_metadata",
            )
            != "true"
        ):
            # Generate metadata from source document
            # Try to get the raw_html or page_content from redis
            source_text = r.hget(redis_key, "raw_html")
            if not source_text:
                source_text = r.hget(redis_key, "page_content")
            if not source_text or len(source_text) < 100:
                return {}
            # Extract metadata from source document using LLM
            try:
                page_metadata = extract_metadata_from_source_document(source_text)
            except OutputParserException:
                return {}

            # Save metadata to redis
            page_metadata_map = clean_up_metadata_object(page_metadata)
            page_metadata_map["fetched_additional_metadata"] = "true"
            r.hset(
                redis_key,
                mapping=page_metadata_map,
            )

    # Return metadata fields from redis
    field_map = dict(
        zip(
            metadata_fields,
            r.hmget(redis_key, *metadata_fields),
        )
    )
    # Filter out None values
    dict_to_return = {k: v for k, v in field_map.items() if v is not None}
    # Convert publishing_date to isoformat
    if "publishing_date" in dict_to_return.keys():
        dict_to_return["publishing_date"] = datetime.datetime.fromtimestamp(
            int(dict_to_return["publishing_date"]), tz=datetime.UTC
        ).isoformat()

    return dict_to_return


def generate_additional_table_context(table: str) -> str:
    """
    Augment a table with additional context to make retrieval more accurate.

    Args:
        table (str): The table to augment as a markdown string.

    Returns:
        str: The augmented table.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from climate_rag.llms import get_chatbot
    from climate_rag.prompts import table_augmentation_prompt

    llm = get_chatbot("gemini-2.5-flash")

    prompt = ChatPromptTemplate.from_messages(
        [("system", table_augmentation_prompt), ("human", "{table}")]
    )

    table_augmenter_chain = prompt | llm | StrOutputParser()
    additional_context = table_augmenter_chain.invoke({"table": table})

    return additional_context


def add_additional_table_context(table: str) -> str:
    """
    Add additional context to a table to make retrieval more accurate.

    Args:
        table (str): The table to augment as a markdown string.

    Returns:
        str: The augmented table.
    """

    return (
        f"<table_context>{generate_additional_table_context(table)}</table_context>\n"
        + table
    )


def initialize_project_indices(r, project_id):
    """
    Initialize Redis search indices for a specific project if they don't exist yet.

    Args:
        r: Redis connection
        project_id: The project ID to initialize indices for

    Returns:
        bool: True if indices were created or already existed, False on error
    """

    project_source_index_name = f"{source_index_name}_{project_id}"
    project_zh_source_index_name = f"{zh_source_index_name}_{project_id}"
    project_ja_source_index_name = f"{ja_source_index_name}_{project_id}"

    # Check which indices exist
    existing_indices = set()
    missing_indices = set()

    for index_name in [
        project_source_index_name,
        project_zh_source_index_name,
        project_ja_source_index_name,
    ]:
        try:
            r.ft(index_name).info()
            existing_indices.add(index_name)
        except ResponseError:
            missing_indices.add(index_name)

    if not missing_indices:
        print(f"All indices already exist for project {project_id}")
        return True

    if existing_indices:
        print(
            f"Found existing indices for project {project_id}: {', '.join(existing_indices)}"
        )
        print(f"Creating missing indices: {', '.join(missing_indices)}")
    else:
        print(
            f"No existing indices found for project {project_id}, creating all indices"
        )

    try:
        # Define the index fields
        schema = (
            TextField("source", sortable=True),
            TextField("key_entity", sortable=True),
            TextField("company_name", sortable=True),
            TextField("title", sortable=True),
            TextField("page_content"),
            NumericField("page_length", sortable=True),
            TagField("type_of_document", sortable=True),
            NumericField("date_added", sortable=True),
            NumericField("publishing_date", sortable=True),
            TagField("fetched_additional_metadata", sortable=True),
            TextField("key_entities"),
            TextField("raw_html"),
            TagField("loader"),
            TagField("project_id", sortable=True),
        )

        # Create indices that don't exist
        for index_name in missing_indices:
            if index_name == project_source_index_name:
                # Create English index
                r.ft(index_name).create_index(
                    schema,
                    definition=IndexDefinition(
                        prefix=[
                            (
                                "climate-rag::source:"
                                if project_id == "langchain"
                                else f"climate-rag::{project_id}::source:"
                            )
                        ],
                        index_type=IndexType.HASH,
                    ),
                )
            elif index_name == project_zh_source_index_name:
                # Create Chinese index with Chinese analyzer
                r.ft(index_name).create_index(
                    schema,
                    definition=IndexDefinition(
                        prefix=[
                            (
                                "climate-rag::source:"
                                if project_id == "langchain"
                                else f"climate-rag::{project_id}::source:"
                            )
                        ],
                        index_type=IndexType.HASH,
                        language="chinese",
                    ),
                )
            elif index_name == project_ja_source_index_name:
                # Create Japanese index with Japanese analyzer
                r.ft(index_name).create_index(
                    schema,
                    definition=IndexDefinition(
                        prefix=[
                            (
                                "climate-rag::source:"
                                if project_id == "langchain"
                                else f"climate-rag::{project_id}::source:"
                            )
                        ],
                        index_type=IndexType.HASH,
                        language="japanese",
                    ),
                )

        # Add project to the list of projects
        r.sadd("climate-rag::projects", project_id)

        print(f"Successfully created all missing indices for project {project_id}")
        return True
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error creating indices for project {project_id}: {e}")
        return False


def transfer_document_between_projects(
    source_uri: str,
    source_project_id: str,
    target_project_id: str,
    r,
    delete_source: bool = False,
) -> bool:
    """
    Transfer (move or copy) a document from one project to another.

    Args:
        source_uri: The URI of the document to transfer
        source_project_id: The source project ID
        target_project_id: The target project ID
        r: Redis connection
        delete_source: If True, the document will be deleted from source after copying

    Returns:
        bool: True if the document was transferred successfully, False otherwise
    """
    from redis import ResponseError

    # Check if document exists in source project
    if len(r.keys(f"climate-rag::{source_project_id}::source:{source_uri}")) > 0:
        source_key = f"climate-rag::{source_project_id}::source:{source_uri}"
    elif len(r.keys(f"climate-rag::source:{source_uri}")) > 0:
        source_key = f"climate-rag::source:{source_uri}"
    else:
        print(f"Document {source_uri} not found in project {source_project_id}")
        return False

    # Check if document already exists in target project
    if len(r.keys(f"climate-rag::{target_project_id}::source:{source_uri}")) > 0:
        print(f"Document {source_uri} already exists in project {target_project_id}")
        return False

    # Get document data from Redis
    try:
        doc_data = r.hgetall(source_key)

        # Update project_id in document
        doc_data["project_id"] = target_project_id

        # Add to target project
        r.hset(
            f"climate-rag::{target_project_id}::source:{source_uri}", mapping=doc_data
        )

        # Don't transfer chroma documents between projects if they are the same
        if source_project_id != target_project_id:
            # Get document from source vector store
            source_db = get_vector_store(source_project_id)
            docs = source_db.get(
                where={"source": {"$in": [source_uri]}},
                include=["documents", "metadatas"],
            )

            if len(docs["ids"]) > 0:
                # Get target vector store
                target_db = get_vector_store(target_project_id)

                # Add to target vector store
                target_db.add_documents(
                    documents=[
                        Document(page_content=doc, metadata=meta)
                        for doc, meta in zip(docs["documents"], docs["metadatas"])
                    ]
                )

                # Delete from source project if requested
                if delete_source:
                    r.delete(source_key)
                    source_db.delete(ids=docs["ids"])

                operation = "moved" if delete_source else "copied"
                print(
                    f"Document {source_uri} {operation} from project {source_project_id} to {target_project_id}"
                )
                return True
            else:
                print(
                    f"Document {source_uri} not found in vector store for project {source_project_id}"
                )
                return False
        else:
            print(
                f"Document {source_uri} already exists in project {target_project_id}"
            )
            return True
    except ResponseError as e:
        print(f"Error transferring document {source_uri}: {e}")
        return False
