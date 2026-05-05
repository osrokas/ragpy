import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

class JsonParser:
    def __init__(self, json_string, splitter: RecursiveCharacterTextSplitter):
        self.data = json.loads(json_string)
        self.splitter = splitter

    def get_document(self):
        return self.data.get("document")

    def get_elements(self):
        document = self.get_document()
        return document.get("elements") if document else None

    def filter_elements_by_type(self, types):
        elements = self.get_elements()
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
    
    def split_text(self):
        document = self.get_document()
        elements = self.filter_elements_by_type(["text", "caption", "figure"])
        content = self.content_creation(elements)
        rows = self.splitter.split_text(content)

        # Add the document ID to each row
        rows_with_id = [{"content": row, "id": index + 1} for index, row in enumerate(rows)]
        return rows_with_id
