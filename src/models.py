from dataclasses import dataclass

@dataclass
class ParsedElement:
    id: int
    content: str

    def to_dict(self):
        return {"id": self.id, "content": self.content}

@dataclass
class ParsedDocument:
    elements: list[ParsedElement]

    def to_dict(self):
        return [element.to_dict() for element in self.elements]
