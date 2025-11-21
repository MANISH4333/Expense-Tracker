from pypdf import PdfReader
from typing import Union
from io import BytesIO

def extract_text_from_pdf(file: Union[str, BytesIO]) -> str:
    """
    Extracts text from a PDF file.

    Args:
        file (Union[str, BytesIO]): A file-like object (e.g., BytesIO) or a string path to the PDF file.

    Returns:
        str: Extracted text from the PDF file.

    Raises:
        Exception: Raises an exception if the PDF cannot be read or there is an extraction issue.
    """
    try:
        reader = PdfReader(file)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

        return ''.join(text)  # Join the list into a single string

    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")
