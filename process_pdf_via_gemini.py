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


def process_pdf_via_gemini(
    pdf_path: str | os.PathLike, document_prefix: str = ""
) -> Tuple[PDFMetadata, str]:
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
    from langchain_core.exceptions import OutputParserException
    import requests
    import mimetypes

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
        response = requests.get(pdf_path)
        mime_type = response.headers["content-type"]
        pdf_data = base64.b64encode(response.content).decode("utf8")
        # Add the source URL to the document prefix
        document_prefix += f"\n\nSource URL: {pdf_path}\n"
    else:
        # If pdf_path is a path to a file, open it and read the contents
        with open(pdf_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode("utf-8")
            mime_type = mimetypes.guess_type(pdf_path)[0]

    metadata_prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": pdf_metadata_extractor_prompt.format(
                            response_format=parser.get_format_instructions()
                        )
                        + "\n"
                        + document_prefix,
                    },
                    {"type": "media", "mime_type": mime_type, "data": pdf_data},
                ]
            ),
        ]
    )
    metadata_extraction_chain = metadata_prompt | llm | parser

    logging.info("Extracting metadata from PDF")
    try:
        pdf_metadata = metadata_extraction_chain.invoke({})
    except OutputParserException as e:
        logging.error(f"Error extracting metadata: {e}")
        # Try one more time
        pdf_metadata = metadata_extraction_chain.invoke({})
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
                    {"type": "media", "mime_type": mime_type, "data": pdf_data},
                ]
            ),
        ],
    )
    llm = get_chatbot("gemini-1.5-flash", max_tokens=8192)
    convert_to_md_chain = convert_prompt | llm
    previous_outputs = []
    num_chunks = pdf_metadata.num_pages // 3 + 1
    list_of_pages = bin_list_into_chunks(
        list(range(1, pdf_metadata.num_pages + 1)), num_chunks
    )
    # Bin pages into chunks
    for nth_chunk, page_range in enumerate(list_of_pages):
        logging.info(f"Converting PDF to markdown: chunk {nth_chunk + 1}/{num_chunks}")
        pdf_md = convert_to_md_chain.invoke(
            {
                "n_pages": pdf_metadata.num_pages,
                "pages_to_return": (
                    f"pages {page_range[0]} to {page_range[-1]}"
                    if len(page_range) > 1
                    else f"page {page_range[0]}"
                ),
            }
        )
        previous_outputs += [
            {"chunk_number": nth_chunk, "chunk_output": pdf_md.content}
        ]
    # Join all chunk outputs
    complete_pdf_md = "\n".join(chunk["chunk_output"] for chunk in previous_outputs)
    logging.info("PDF converted to markdown")

    # Check that we have all pages in the markdown
    page_exists = [
        f"Page {page_no}" in complete_pdf_md
        for page_no in range(1, pdf_metadata.num_pages + 1)
    ]
    if False in page_exists:
        logging.warning("Not all pages are present in the markdown output")
    # If there is only one page, remove the page number from the markdown
    if pdf_metadata.num_pages == 1:
        complete_pdf_md = complete_pdf_md.replace("## Page 1 of 1\n", "")

    # Add the document prefix to the markdown
    complete_pdf_md = document_prefix + "\n" + complete_pdf_md

    return pdf_metadata, complete_pdf_md


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a PDF file using a Gemini chatbot."
    )
    parser.add_argument(
        "pdf_path", type=str, help="The path to the PDF file to process."
    )
    parser.add_argument(
        "--document-prefix",
        type=str,
        default="",
        help="Prefix to add to the document when parsing. Useful for adding context to the document.",
    )
    args = parser.parse_args()

    pdf_metadata, pdf_md = process_pdf_via_gemini(args.pdf_path, args.document_prefix)
    console = Console()
    console.print(pdf_metadata)
    console.print(Markdown(pdf_md))
