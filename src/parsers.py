import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFium2Loader


class JsonParser:
    def __init__(self, data: dict, splitter: RecursiveCharacterTextSplitter):
        self.data = data
        self.splitter = splitter

    @classmethod
    def from_string(cls, json_string, splitter):
        data = json.loads(json_string)
        return cls(data, splitter)
    
    @classmethod
    def from_strings(cls, json_strings:list, splitter):
        data = cls.merge_documents(json_strings)
    
        return cls(data, splitter)
    
    @staticmethod
    def merge_documents(json_strings:list):
        merged_elements = []
        for j_string in json_strings:
            doc = json.loads(j_string)
            merged_elements.extend(doc["document"]["elements"])
        return {"document": {"elements": merged_elements}}
        
    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return json.dumps(self.data, indent=4)

    def _get_document(self):
        return self.data.get("document")

    def _get_elements(self):
        document = self._get_document()
        return document.get("elements") if document else None

    def _filter_elements_by_type(self, types):
        elements = self._get_elements()
        if elements is not None:
            return [el for el in elements if el.get("type") in types]
        return None
    
    @staticmethod
    def content_creation(content):
        content_data = ''
        for row in content:
            if row["description"] == None:
                text = row["content"]
            else:
                text = row["description"]
            content_data += text + " "
        return content_data
    
    def split_text(self, types: list):
        rows = []
        elements = self._filter_elements_by_type(types)
        content = self.content_creation(elements)
        rows = self.splitter.split_text(content)

        # Add the document ID to each row
        rows_with_id = [{"content": row, "id": index + 1} for index, row in enumerate(rows)]
        return rows_with_id


class DocumentProcessor:
    def __init__(self, path: str, splitter: RecursiveCharacterTextSplitter):
        self.path = path
        self.loader = PyPDFium2Loader(self.path)
        self.splitter = splitter

    def create_content(self):
        docs = self.loader.load()
        content = ''
        for doc in docs:
            doc.page_content = doc.page_content.replace('\n', ' ')
            content += doc.page_content
        return content
    
    def split_text(self):
        content = self.create_content()
        rows = self.splitter.split_text(content)

        # Add the document ID to each row
        rows_with_id = [{"content": row, "id": index + 1} for index, row in enumerate(rows)]
        return rows_with_id