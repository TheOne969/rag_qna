import pymupdf4llm

def extract_text_from_pdf(pdf_path, max_pages=None):
    """Yield text and page number from the given PDF file as Markdown."""
    # to_markdown with page_chunks=True handles layout & automatic OCR page-by-page
    pages_data = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    
    for page_num, page_dict in enumerate(pages_data):
        if max_pages is not None and page_num >= max_pages:
            break
        yield {
            "page": page_num + 1,
            "text": page_dict["text"], 
            "filename": pdf_path.split("/")[-1],
        }

def extract_text_as_documents(pdf_path, max_pages=None):
    return [item["text"] for item in extract_text_from_pdf(pdf_path, max_pages=max_pages)]

if __name__ == "__main__":
    pdf_path = "sample.pdf"
    for page_data in extract_text_from_pdf(pdf_path, max_pages=10):
        print(f"Page {page_data['page']}: {page_data['text'][:1000]}...\n")