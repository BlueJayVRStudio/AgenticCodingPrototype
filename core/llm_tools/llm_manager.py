from openai import OpenAI
from langchain_openai import ChatOpenAI
from core.config.settings_loader import Settings
import time

class LLMManager:
    """
    Centralized LLM setup for both:
      - LangChain Chat LLM (ChatOpenAI)
      - Raw OpenAI client for chat completions
    """

    def __init__(self, settings: Settings, agent_name: str):
        self.agent_name = agent_name
        self.settings = settings
        self.agent_conf = self.settings.load_agent_config(agent_name)

        llm_conf = self.agent_conf["llm"]["chat"]
        comp_conf = self.agent_conf["llm"]["chat_completion"]

        self.llm_api_key = settings.resolve_api_key(llm_conf["api_key_name"])
        self.comp_api_key = settings.resolve_api_key(comp_conf["api_key_name"])

        # Store configurations
        self.chat_model = llm_conf["model"]
        self.comp_model = comp_conf["model"]
        self.chat_base = llm_conf["base_url"]
        self.comp_base = comp_conf["base_url"]

        # LangChain-compatible Chat LLM
        self.chat_llm = ChatOpenAI(
            model=self.chat_model,
            openai_api_base=self.chat_base,
            openai_api_key=self.llm_api_key,
        )

        # Raw client (for completions or advanced calls)
        self.client = OpenAI(
            base_url=self.comp_base,
            api_key=self.comp_api_key,
        )

        self.rate_limit_chat_completions = self.agent_conf["llm"]["chat_completion"]["rate_limit_seconds"]

    # Public Accessors
    def get_chat_llm(self):
        """Return LangChain Chat LLM instance."""
        return self.chat_llm

    def get_client(self):
        """Return raw OpenAI client instance."""
        return self.client

    # Chat Completion Helpers
    def chat_completion(
        self,
        messages: list,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        model: str = None,
    ) -> str:
        """
        Perform a chat completion request using the raw OpenAI client.
        """
        time.sleep(self.rate_limit_chat_completions)

        response = self.client.chat.completions.create(
            model=model or self.comp_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response

    def structured_chat(
        self,
        system_prompt: str,
        user_query: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        model: str = None,
    ) -> str:
        """
        A simpler helper for single-turn chats — provides a quick interface for
        instruction + query pair completions.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        return self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
