import logging
import shutil
from typing import List, Optional
import msgspec
from pathvalidate import sanitize_filename as _sanitize_filename
import os
import pdf2docx
import urllib.parse
import re
import pandas as pd
from markdown_pdf import Section, MarkdownPdf

from cache import r

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing special characters.

    Args:
        filename (str): The filename to sanitize.

    Returns:
        str: The sanitized filename.
    """
    return _sanitize_filename(filename).replace(" ", "_")

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
def md_to_pdf(md_string: str, pdf_path: str) -> str:
    """
    Convert a markdown string to a PDF file.

    Args:
        md_string (str): The markdown string to convert
        pdf_path (str): The path to save the PDF file to
    """
    # Create a new PDF document
    pdf = MarkdownPdf(toc_level=2)
    # Add a section to the PDF
    pdf.add_section(Section(md_string))
    # Write the PDF to disk
    pdf.save(pdf_path)

    return pdf_path

def pdf_to_docx(pdf_path: str, docx_path: str) -> str:
    """
    Convert a PDF file to a DOCX file.

    Args:
        pdf_path (str): The path to the PDF file to convert
        docx_path (str): The path to save the DOCX file to
    """
    pdf2docx.Converter(pdf_path).convert(docx_path)

    return docx_path

def sanitize_url(url):
    domain_part, filename_part = url.rsplit("/", 1)

    return domain_part + "/" + urllib.parse.quote(filename_part)


def clean_urls(urls, static_path=""):
    """
    Clean a list of URLs by removing irrelevant domain names.

    Args:
        urls (list): A list of URLs to clean.

    Returns:
        list: A list of cleaned URLs.
    """
    # Convert to pandas series
    urls = pd.Series(urls)

    # Remove tmpfiles.org prefix
    pattern = r"https://r\.jina\.ai/https://tmpfiles\.org/dl/.+?#"
    urls = urls.str.replace(pattern, static_path, regex=True)
    # Remove jina.ai prefix
    urls = urls.str.replace("https://r.jina.ai/", "")
    return urls.tolist()

def generate_qa_id(question: str, answer: str) -> str:
    """
    Return the question/answer ID from a given answer string by calculating the hash of the answer string and appending it to the question.

    Args:
        question (str): The question string.
        answer (str): The answer string.

    Returns:
        str: The question/answer ID.
    """
    import hashlib

    question_string = question[:200]
    answer_hash = hashlib.shake_256(answer.encode()).hexdigest(5)
    qa_id = sanitize_filename(f"{question_string}_{answer_hash}")

    return qa_id

def compile_answer(generation: str, initial_question: str, sources: List[str | None]) -> str:
    """
    Compile the answer from the generation and the sources.

    Args:
        generation (str): The generated answer.
        documents (List[GraphState]): The list of documents.

    Returns:
        str: The compiled answer.
    """
    answer = (
        f"""# {initial_question}\n\n"""
        + generation
        + "\n\n**Sources:**\n\n"
        + "\n\n".join(
            set(
                [
                    (" * " + clean_urls([source], os.environ.get("STATIC_PATH", ""))[0])
                    for source in sources
                    if source is not None
                ]
            )
        )
    )

    return answer

def render_qa_pdfs(qa_id):
    from helpers import md_to_pdf, pdf_to_docx
    filename = sanitize_filename(qa_id)
    qa_map = r.hgetall(f"climate-rag::answer:{qa_id}")
    # Check if PDF is already in redis cache
    if qa_map.get("pdf_uri", None) is not None:
        pdf_download_url = qa_map["pdf_uri"]
        docx_download_url = qa_map["docx_uri"]
    else:
        print(qa_id)
        answer = compile_answer(
            qa_map["answer"], qa_map["question"], msgspec.json.decode(qa_map["sources"])
        )
        os.makedirs("tmp", exist_ok=True)
        pdf_path = f"tmp/{filename}.pdf"
        docx_path = f"tmp/{filename}.docx"

        md_to_pdf(answer, pdf_path)
        pdf_to_docx(pdf_path, docx_path)

        STATIC_PATH = os.environ.get("STATIC_PATH", "")
        UPLOAD_FILE_PATH = os.environ.get("UPLOAD_FILE_PATH", "")
        USE_S3 = os.environ.get("USE_S3", False) == "True"

        if (STATIC_PATH != "") and (UPLOAD_FILE_PATH != ""):
            # Copy the files to the static path
            os.makedirs(f"{UPLOAD_FILE_PATH}/outputs", exist_ok=True)
            shutil.copy(pdf_path, f"{UPLOAD_FILE_PATH}/outputs/{filename}.pdf")
            shutil.copy(docx_path, f"{UPLOAD_FILE_PATH}/outputs/{filename}.docx")
            # Serve the files from the static path instead
            pdf_download_url = f"{STATIC_PATH}/outputs/{filename}.pdf"
            docx_download_url = f"{STATIC_PATH}/outputs/{filename}.docx"
        elif (STATIC_PATH != "") and (USE_S3 == True):
            # Upload the files to S3
            if not upload_file(
                file_name=pdf_path,
                bucket=os.environ.get("S3_BUCKET", ""),
                path="/outputs/",
                object_name=f"{filename}.pdf",
            ):
                logging.error(f"Failed to upload {pdf_path} to S3")
            if not upload_file(
                file_name=docx_path,
                bucket=os.environ.get("S3_BUCKET", ""),
                path="/outputs/",
                object_name=f"{filename}.docx",
            ):
                logging.error(f"Failed to upload {docx_path} to S3")
            # Serve the files from S3
            pdf_download_url = f"{STATIC_PATH}/outputs/{filename}.pdf"
            docx_download_url = f"{STATIC_PATH}/outputs/{filename}.docx"
        else:
            pdf_download_url = pdf_path
            docx_download_url = docx_path

        # Save PDF and DOCX locations to redis cache
        r.hset("climate-rag::answer:" + qa_id, "pdf_uri", pdf_download_url)
        r.hset("climate-rag::answer:" + qa_id, "docx_uri", docx_download_url)
    return pdf_download_url,docx_download_url


def modify_document_source_urls(old_url, new_url, db, r):
    from redis import ResponseError
    # Rename redis key
    try:
        r.rename(f"climate-rag::source:{old_url}", f"climate-rag::source:{new_url}")
        r.hset(f"climate-rag::source:{new_url}", "source", new_url)
        r.hset(f"climate-rag::source:{new_url}", "source_alt", old_url)
    except ResponseError:
        print(f"Redis key not found for source: {old_url}")
    docs = db.get(where={"source": {"$in": [old_url]}}, include=["metadatas"])
    if len(docs["ids"]) > 0:
        for doc in docs["metadatas"]:
            doc["source_alt"] = doc["source"]
            doc["source"] = new_url

        db._collection.update(ids=docs["ids"], metadatas=docs["metadatas"])
    else:
        print(f"No chroma documents found with source: {old_url}")