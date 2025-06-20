from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_texts(pages: list[str], chunk_size=512, chunk_overlap=64):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    page_metas = [{"page": i + 1} for i in range(len(pages))]
    return splitter.create_documents(pages, metadatas=page_metas)

""" 

We are going for character-wise chunking for now. 
Later on, we could try sentence wise or some other chunking as an experiment

Still, it's not ready for column-wise or any other weirdly formatted pdf. 
"""