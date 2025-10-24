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
    def __init__(self, settings: Settings, agent_name: str):
        self.agent_conf = settings.load_agent_config(agent_name)

        embedding_conf = self.agent_conf["memory"]["embedding"]
        provider = embedding_conf["provider"]
        model = embedding_conf["model"]
        api_key_name = embedding_conf["api_key_name"]
        self.embeddings = None
        
        if provider == "OLLAMA":
            self.embeddings = OllamaEmbeddings(model=model)
        elif provider == "FIREWORKS":
            self.embeddings = FireworksEmbeddings(
                model=model,
                fireworks_api_key=os.getenv(api_key_name)
            )
        elif "@" in provider:
            self.embeddings = OllamaEmbeddings(
                base_url= provider.split("@")[1],
                model=model
            )

    def get_provider(self):
        return self.embeddings