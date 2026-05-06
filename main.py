from fastapi import FastAPI

from src.retriever import Retriever


app = FastAPI()
retriever = Retriever()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/rag")
def rag(query: str):
    answer = retriever.run(query)
    return {"query": query, "answer": answer}
