from dataclasses import dataclass

@dataclass
class DeltaIndexRAG:
    index_name: str
    columns: str
    primary_key: str
    embedding_model: str
