import os
import argparse

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.parsers import DocumentProcessor
from src.embedings import embedding_factory


def main(pdf_path: str, directory: str = "./chroma_db", embedding_client: str = "huggingface"):
    """Create a Chroma vector store from a PDF document."""
    # Initialize text splitter 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Document processing
    document_processor = DocumentProcessor(pdf_path, text_splitter)
    data = document_processor.split_text()
    text_chunks = [chunk['content'] for chunk in data]
    print(f"Split text into {len(data)} chunks")
    print("Creating embeddings...")

    # Define embedding model
    embeddings = embedding_factory(embedding_client=embedding_client)

    print("Embeddings created successfully.")
    print("Creating vector store...")

    # Create Chroma vector store from text chunks and embeddings
    Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory=directory
    )
    print("Vector store created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a vector store from a PDF document.")
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF document.")
    parser.add_argument("--directory", type=str, default="./chroma_db", help="Directory to store the Chroma vector store.")
    parser.add_argument("--embedding_client", type=str, default="huggingface", help="Embedding client to use.")
    args = parser.parse_args()
    pdf_path = args.pdf_path
    directory = args.directory
    embedding_client = args.embedding_client

    if os.path.exists(pdf_path):
        print(f"PDF file found at: {pdf_path}")
    else:
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    print(f"Processing PDF file: {pdf_path}")
    main(pdf_path, directory, embedding_client)