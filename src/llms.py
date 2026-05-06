from ollama import chat
from langchain_core.language_models.llms import LLM

class OllamaLLM(LLM):
    model : str = "ministral-3:14b-cloud"

    def _call(self, prompt: str, stop=None):
        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content
    
    @property
    def _llm_type(self):
        return "ollama"