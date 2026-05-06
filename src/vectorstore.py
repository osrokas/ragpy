from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = "./chroma_db"


class VectorStore:
    def __init__(self, chunks, embeddings, directory=CHROMA_DIR, k=4):
        self.chunks = chunks
        self.vectorstore = Chroma.from_texts(
            texts=self.chunks,
            embedding=embeddings,
            persist_directory=directory
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

    def get_retriever(self):
        return self.retriever