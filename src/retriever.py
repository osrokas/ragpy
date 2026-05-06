from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings
CHROMA_DIR = "./chroma_db"

embeddings = HuggingFaceEmbeddings()


vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# data = vectorstore.similarity_search("Djikstra", k=4)

# for row in data:
#     print(row.metadata)
#     print(row.page_content[:300])
#     # print("----")
from langchain_core.language_models.llms import LLM
from ollama import chat

class OllamaCloudLLM(LLM):
    model: str = "ministral-3:14b-cloud"

    def _call(self, prompt: str, stop=None):
        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content

    @property
    def _llm_type(self):
        return "ollama_cloud"

from langchain_classic.chains import RetrievalQA

llm = OllamaCloudLLM()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

print(qa_chain.run("What is Dijkstra's algorithm? What data source did you use to answer this question?"))
