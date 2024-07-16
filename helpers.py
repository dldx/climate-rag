import pdf2docx
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
    import re
    s = str(name).strip().replace(" ", "_")
    s = re.sub(r"(?u)[^-\w.]", "", s)
    if s in {"", ".", ".."}:
        raise AssertionError("Could not derive file name from '%s'" % name)
    return s