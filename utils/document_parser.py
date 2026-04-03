import io
import logging
import fitz  # PyMuPDF
from docx import Document
from fastapi import UploadFile, HTTPException

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    text = ""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise ValueError(f"Could not extract text from PDF: {str(e)}")

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        doc = Document(io.BytesIO(file_bytes))
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text).strip()
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        raise ValueError(f"Could not extract text from DOCX: {str(e)}")

async def parse_resume(file: UploadFile) -> str:
    """Entry point for parsing resume files (PDF or DOCX)."""
    content = await file.read()
    filename = file.filename.lower() if file.filename else ""
    
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(content)
    elif filename.endswith((".docx", ".doc")):
        return extract_text_from_docx(content)
    else:
        # Try PDF as fallback if no extension
        try:
            return extract_text_from_pdf(content)
        except:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload a PDF or DOCX file."
            )
