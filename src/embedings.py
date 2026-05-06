from langchain.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

def embedding_factory(embedding_client: str = "huggingface") -> Embeddings:
    if embedding_client == "huggingface":
        return HuggingFaceEmbeddings()
    else:
        raise ValueError(f"Unsupported embedding client: {embedding_client}")

