from core.config.settings_loader import *
from agents.base_agent import *
from core.embedding_tools.embedding_provider import EmbeddingProvider
from core.db_tools.vector_db_provider import VectorDBProvider
from core.llm_tools.llm_chat_provider import LLMChatProvider
from core.llm_tools.llm_chat_completion_provider import LLMChatCompletionProvider

class AgentFactory:
    def __init__(self, settings=None):
        self.settings = settings or Settings()

    def create_base_agent(self, agent_name: str):
        project_root_provider = ProjectRootProvider(self.settings, agent_name)
        embedding_provider = EmbeddingProvider(self.settings, agent_name)
        vector_db_provider = VectorDBProvider(self.settings, agent_name, embedding_provider)
        llm_chat_provider = LLMChatProvider(self.settings, agent_name)
        llm_chat_completion_provider = LLMChatCompletionProvider(self.settings, agent_name)
        
        return BaseAgent(
            project_root_provider, 
            vector_db_provider, 
            llm_chat_provider, 
            llm_chat_completion_provider, 
            agent_name
        )

    def hybridize_base_agent(
        self, 
        project_root_provider_config: str,
        embedding_provider_config: str,
        vector_db_provider_config: str,
        llm_chat_provider_config: str,
        llm_chat_completion_provider_config: str
    ):
        project_root_provider = ProjectRootProvider(self.settings, project_root_provider_config)
        embedding_provider = EmbeddingProvider(self.settings, embedding_provider_config)
        vector_db_provider = VectorDBProvider(self.settings, vector_db_provider_config, embedding_provider)
        llm_chat_provider = LLMChatProvider(self.settings, llm_chat_provider_config)
        llm_chat_completion_provider = LLMChatCompletionProvider(self.settings, llm_chat_completion_provider_config)

        return BaseAgent(
            project_root_provider, 
            vector_db_provider, 
            llm_chat_provider, 
            llm_chat_completion_provider, 
            "hybrid_agent"
        )