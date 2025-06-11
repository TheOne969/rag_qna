import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path, max_pages=None):
    """Yield text and page number from the given PDF file."""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):    
        if max_pages is not None and page_num >= max_pages:
            break
        page = doc[page_num]
        text = page.get_text("text")
        yield {
            "page": page_num + 1,
            "text": text,
            "filename": pdf_path.split("/")[-1],
        }


def extract_text_as_documents(pdf_path, max_pages=None):
    return [item["text"] for item in extract_text_from_pdf(pdf_path, max_pages=max_pages)]


if __name__ == "__main__":
    pdf_path = "sample.pdf"
    for page_data in extract_text_from_pdf(pdf_path, max_pages=3):
        print(f"Page {page_data['page']}: {page_data['text'][:200]}...\n")
        
"""
This file is individually working properly. have some formatting errors with column wise pdf
But it remains to be see if that would be a problem later. 
"""