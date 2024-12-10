import logging
import os
from typing import Tuple
from llms import get_chatbot
import base64
from schemas import PDFMetadata
import argparse
from rich.console import Console
from rich.markdown import Markdown

logging.basicConfig(level=logging.INFO)


def process_pdf_via_gemini(pdf_path: str | os.PathLike) -> Tuple[PDFMetadata, str]:
    """
    Process a PDF file using a Gemini flash chatbot.

    Args:
        pdf_path (str): The path to the PDF file to process.
        llm (str): The language model to use. Options are "gemini-1.5-pro", "gemini-1.5-flash".

    Returns:
        PDFMetadata: The extracted metadata from the PDF.
        str: The PDF content converted to markdown.
    """


    from langchain_core.prompts import ChatPromptTemplate
    import requests

    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.messages import HumanMessage
    from prompts import pdf_metadata_extractor_prompt, convert_to_md_prompt
    from helpers import bin_list_into_chunks

    llm = get_chatbot("gemini-1.5-flash")
    # First extract the metadata from the PDF
    parser = PydanticOutputParser(pydantic_object=PDFMetadata)
    # If pdf_path is a url, download the file and read the contents
    if pdf_path.startswith("http"):
        pdf_data = base64.b64encode(requests.get(pdf_path).content).decode("utf8")
    else:
        # If pdf_path is a path to a file, open it and read the contents
        with open(pdf_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode("utf-8")

    metadata_prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": pdf_metadata_extractor_prompt.format(
                            response_format=parser.get_format_instructions()
                        ),
                    },
                    {"type": "media", "mime_type": "application/pdf", "data": pdf_data},
                ]
            ),
        ]
    )
    reader_chain = metadata_prompt | llm | parser

    logging.info("Extracting metadata from PDF")
    pdf_metadata = reader_chain.invoke({})
    if pdf_metadata.num_pages == 0:
        raise ValueError("The PDF has 0 pages")
    if pdf_metadata.scanned_pdf == False:
        logging.warning("The PDF is not a scanned document.")
    convert_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                convert_to_md_prompt,
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Follow the system prompt"},
                    {"type": "media", "mime_type": "application/pdf", "data": pdf_data},
                ]
            ),
        ],
    )
    llm = get_chatbot("gemini-1.5-flash", max_tokens=8192)
    convert_to_md_chain = convert_prompt | llm
    previous_outputs = []
    num_chunks = pdf_metadata.num_pages // 3 + 1
    list_of_pages = bin_list_into_chunks(list(range(1, pdf_metadata.num_pages + 1)), num_chunks)
    # Bin pages into chunks
    for nth_chunk, page_range in enumerate(list_of_pages):
        logging.info(f"Converting PDF to markdown: chunk {nth_chunk + 1}/{num_chunks}")
        pdf_md = convert_to_md_chain.invoke(
            {
                "n_pages": pdf_metadata.num_pages,
                "pages_to_return": f"pages {page_range[0]} to {page_range[-1]}" if len(page_range) > 1 else f"page {page_range[0]}",
            }
        )
        previous_outputs += [
            {"chunk_number": nth_chunk, "chunk_output": pdf_md.content}
        ]
    # Join all chunk outputs
    complete_pdf_md = "\n".join(chunk["chunk_output"] for chunk in previous_outputs)
    logging.info("PDF converted to markdown")

    # Check that we have all pages in the markdown
    page_exists = [f"Page {page_no}" in complete_pdf_md for page_no in range(1, pdf_metadata.num_pages+1)]
    if False in page_exists:
        logging.warning("Not all pages are present in the markdown output")

    return pdf_metadata, complete_pdf_md


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a PDF file using a Gemini chatbot."
    )
    parser.add_argument(
        "pdf_path", type=str, help="The path to the PDF file to process."
    )
    args = parser.parse_args()

    pdf_metadata, pdf_md = process_pdf_via_gemini(args.pdf_path)
    console = Console()
    console.print(pdf_metadata)
    console.print(Markdown(pdf_md))
