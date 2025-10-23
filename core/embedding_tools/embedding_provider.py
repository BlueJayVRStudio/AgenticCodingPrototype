import os
from core.config.settings_loader import Settings
from langchain_fireworks import FireworksEmbeddings
from langchain.embeddings import OllamaEmbeddings

class EmbeddingProvider:
    """
    Factory for creating embedding model instances based on the specified provider.

    This class abstracts the initialization logic for different embedding backends.
    It currently supports the following providers:
      - OLLAMA
      - FIREWORKS

    The design allows easy extension to support additional providers in the future.
    """
    def __init__(self, provider: str, model: str, api_key_name: str):
        self.embeddings = None
        if provider == "OLLAMA":
            self.embeddings = OllamaEmbeddings(model=model)
        elif provider == "FIREWORKS":
            self.embeddings = FireworksEmbeddings(
                model=model,
                fireworks_api_key=os.getenv(api_key_name)
            )
    
    def get_provider(self):
        return self.embeddings