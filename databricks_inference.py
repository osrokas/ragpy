from databricks.vectorstore import DatabricksVectorSearchStore, DatabricksRAG
import os

if __name__ == "__main__":

    vector_search_endpoint_name = "my_vector_search_endpoint"
    index_name = "my_index"
    text_column = "content"
    source_table_name = "rag.default.texts"
    index_name = "rag.default.embeddings_index"
    embedding_endpoint = "databricks-gte-large-en"

    endpoint = DatabricksVectorSearchStore(
        endpoint_name=vector_search_endpoint_name,
        source_table_name=source_table_name,
        index_name=index_name,
        text_column=text_column,
        embedding_endpoint=embedding_endpoint,
        workspace_url=os.getenv("DATABRICKS_HOST"),
        personal_access_token=os.getenv("DATABRICKS_TOKEN")
    )
    endpoint.create()
    db_rag = DatabricksRAG(
        vector_search_store=endpoint,
        system_prompt="You are a helpful assistant for answering questions about the documents in the database. Use only the provided context to answer the question. If you don't know the answer, say you don't know, don't try to make up an answer."
    )
    question = {"query": "What is graph data structure?"}
    answer = db_rag.run(question)
    print(answer)
