from openai import OpenAI
from langchain_openai import ChatOpenAI
from core.config.settings_loader import Settings
import time

class LLMChatProvider:
    """
    """

    def __init__(self, settings: Settings, agent_name: str):
        self.agent_name = agent_name
        self.settings = settings
        self.agent_conf = self.settings.load_agent_config(agent_name)

        llm_conf = self.agent_conf["llm"]["chat"]
        self.llm_api_key = settings.resolve_api_key(llm_conf["api_key_name"])

        # Store configurations
        self.chat_model = llm_conf["model"]
        self.chat_base = llm_conf["base_url"]

        # LangChain-compatible Chat LLM
        self.chat_llm = ChatOpenAI(
            model=self.chat_model,
            openai_api_base=self.chat_base,
            openai_api_key=self.llm_api_key,
        )

        self.rate_limit_chat_completions = self.agent_conf["llm"]["chat_completion"]["rate_limit_seconds"]

    # Public Accessors
    def get_chat_llm(self):
        """Return LangChain Chat LLM instance."""
        return self.chat_llm

