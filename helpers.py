import pdf2docx
import re
import pandas as pd
from markdown_pdf import Section, MarkdownPdf

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

def get_valid_filename(name):
    s = str(name).strip().replace(" ", "_")
    s = re.sub(r"(?u)[^-\w.]", "", s)
    if s in {"", ".", ".."}:
        raise AssertionError("Could not derive file name from '%s'" % name)
    return s

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

    answer_hash = hashlib.shake_256(answer.encode()).hexdigest(5)
    qa_id = f"{question[:200]}_{answer_hash}"

    return qa_id