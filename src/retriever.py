from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

from src.llms import OllamaLLM
from src.embedings import embedding_factory

CHROMA_DIR = "./chroma_db"


class Retriever():
    def __init__(self, embeding_client: str = "huggingface", llm_model: str = "ministral-3:3b-cloud", chroma_dir: str = CHROMA_DIR):
        self.vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embedding_factory(embedding_client=embeding_client)
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self.llm = OllamaLLM(model=llm_model)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever
        )

    def run(self, query: str):
        return self.qa_chain.run(query)
