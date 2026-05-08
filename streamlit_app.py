import os

import streamlit as st

from ragpy_modules.vectorstore import DatabricksVectorSearchStore, DatabricksRAG

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant for answering questions about the documents in "
    "the database. Use only the provided context to answer the question. If you "
    "don't know the answer, say you don't know, don't try to make up an answer."
)


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return value


@st.cache_resource(show_spinner=False)
def build_rag_client(
    vector_search_endpoint_name: str,
    index_name: str,
    text_column: str,
    source_table_name: str,
    embedding_endpoint: str,
    llm_endpoint: str,
    max_tokens: int,
    system_prompt: str,
) -> DatabricksRAG:
    host = _require_env("DATABRICKS_HOST")
    token = _require_env("DATABRICKS_TOKEN")

    endpoint = DatabricksVectorSearchStore(
        endpoint_name=vector_search_endpoint_name,
        source_table_name=source_table_name,
        index_name=index_name,
        text_column=text_column,
        embedding_endpoint=embedding_endpoint,
        workspace_url=host,
        personal_access_token=token,
    )
    endpoint.create()

    return DatabricksRAG(
        vector_search_store=endpoint,
        system_prompt=system_prompt,
        llm_endpoint=llm_endpoint,
        max_tokens=max_tokens,
    )


def main() -> None:
    st.set_page_config(page_title="Databricks RAG Chat", page_icon="💬", layout="wide")
    st.title("Databricks RAG Chat")
    st.caption("Chat with your Databricks Vector Search-backed RAG assistant")

    with st.sidebar:
        st.header("Configuration")
        vector_search_endpoint_name = st.text_input(
            "Vector Search Endpoint", value="my_vector_search_endpoint"
        )
        index_name = st.text_input("Index Name", value="rag.default.embeddings_index")
        text_column = st.text_input("Text Column", value="content")
        source_table_name = st.text_input("Source Table", value="rag.default.texts")
        embedding_endpoint = st.text_input(
            "Embedding Endpoint", value="databricks-gte-large-en"
        )
        llm_endpoint = st.text_input("LLM Endpoint", value="databricks-llama-4-maverick")
        max_tokens = st.number_input("Max Tokens", min_value=32, max_value=4096, value=200)
        system_prompt = st.text_area("System Prompt", value=DEFAULT_SYSTEM_PROMPT, height=140)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your indexed documents...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    rag = build_rag_client(
                        vector_search_endpoint_name=vector_search_endpoint_name,
                        index_name=index_name,
                        text_column=text_column,
                        source_table_name=source_table_name,
                        embedding_endpoint=embedding_endpoint,
                        llm_endpoint=llm_endpoint,
                        max_tokens=int(max_tokens),
                        system_prompt=system_prompt,
                    )
                    answer = rag.run(question)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:
                error_message = f"Inference failed: {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )


if __name__ == "__main__":
    main()
