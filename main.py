import os

from langchain_community.embeddings import HuggingFaceEmbeddings

from src.vectorstore import VectorStore
from src.parsers import RecursiveCharacterTextSplitter, DocumentProcessor

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    pdf_path = os.path.join(data_dir, "grokking_algorithms.pdf")
    if os.path.exists(pdf_path):
        print(f"PDF file found at: {pdf_path}")
    else:
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    print(f"Processing PDF file: {pdf_path}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_processor = DocumentProcessor(pdf_path, text_splitter)

    data = document_processor.split_text()

    text_chunks = [chunk['content'] for chunk in data]
    print(f"Split text into {len(data)} chunks")

    print("Creating embeddings...")

    embeddings = HuggingFaceEmbeddings()
    print("Embeddings created successfully.")
    print("Creating vector store...")
    vector_store = VectorStore(text_chunks, embeddings)
    print("Vector store created successfully.")
    vector_store.get_retriever()
    # document_processor.create_csontent()
    # parser = DocumentParser(pdf_path)
    # document = parser.get_document()
    # elements = parser.filter_elements_by_type(["text", "caption", "figure"])

if __name__ == "__main__":
    main()
