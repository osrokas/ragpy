from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks
from databricks.vector_search.client import VectorSearchClient
from langchain_community.embeddings import DatabricksEmbeddings
from langchain_community.vectorstores import DatabricksVectorSearch

from .models import DeltaIndexRAG


class DatabricksVectorSearchStore(VectorSearchClient):
    """Class to manage Databricks Vector Search endpoints and indexes."""
    
    def __init__(self, endpoint_name: str, index_name: str, text_column: str, embedding_endpoint: str, source_table_name: str = "", **vsc_kwargs):
        self.endpoint_name = endpoint_name
        self.index_name = index_name
        self.text_column = text_column
        self.source_table_name = source_table_name
        super().__init__(**vsc_kwargs)
        self._embedding_model = DatabricksEmbeddings(endpoint=embedding_endpoint)
        self._index_rag = DeltaIndexRAG(
            index_name=index_name,
            columns=text_column,
            primary_key="id",
            embedding_model=embedding_endpoint
        )

    def _create_endpoint(self, endpoint_type: str = "STANDARD"):
        """Create a Databricks Vector Search endpoint if it doesn't already exist."""
        if self.endpoint_exists(name=self.endpoint_name):
            print(f"Endpoint '{self.endpoint_name}' already exists.")
        else:
            self.create_endpoint(name=self.endpoint_name, endpoint_type=endpoint_type)

    def _create_index(self):
        """Create a Databricks Vector Search index if it doesn't already exist."""
        if self.index_exists(endpoint_name=self.endpoint_name, index_name=self.index_name):
            print(f"Index '{self.index_name}' already exists on endpoint '{self.endpoint_name}'.")
        else:
            return self.create_delta_sync_index(endpoint_name=self.endpoint_name, index_name=self._index_rag.index_name, source_table_name=self.source_table_name,
                                                  columns_to_sync=self._index_rag.columns, primary_key=self._index_rag.primary_key, embedding_source_column=self.text_column
                                                , pipeline_type="TRIGGERED", embedding_model_endpoint_name=self._index_rag.embedding_model)

    def _get_index(self):
        return self.get_index(endpoint_name=self.endpoint_name, index_name=self.index_name)
    
    def create(self):
        print(f"Creating endpoint '{self.endpoint_name}'")
        self._create_endpoint()
        print(f"Creating index '{self.index_name}'")
        self._create_index()
        # Create the retriever
        print(f"Create Vector Search retriever for index '{self.index_name}'")

    def get_retriever(self):
        vectorestore = DatabricksVectorSearch(
            self._get_index(), text_column=self.text_column, embedding=self._embedding_model
        )

        return vectorestore.as_retriever()
    
class DatabricksRAG:
    """Class to manage a RAG system using Databricks Vector Search."""
    
    def __init__(self, vector_search_store: DatabricksVectorSearchStore, system_prompt: str, llm_endpoint: str = "databricks-llama-4-maverick", max_tokens: int = 200):
        self.vector_search_store = vector_search_store
        self.retriever = vector_search_store.get_retriever()
        self.llm = ChatDatabricks(target_uri="databricks", endpoint=llm_endpoint, max_tokens=max_tokens)
        self.system_prompt = system_prompt
        
        template = f"""{system_prompt}
            Use the following pieces of context to answer the question at the end:
            {{context}}
            Question: {{question}}
            Answer:
            """
        self.prompt = PromptTemplate(template=template  , input_variables=["system_prompt", "context", "question"])
        self.chain = self._create_chain()

    def _create_chain(self):
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
        )
        return chain
    
    def run(self, query: str):
        return self.chain.run(query)