import os
import yaml
from pathlib import Path
from dotenv import load_dotenv


class Settings:
    """
    Centralized configuration loader for AgenticAI.
    Supports both:
      - Global config (settings.yaml)
      - Agent-specific configs (agents/<agent_name>.yaml)
    """

    def __init__(self, global_path: str = None):
        # Always load global settings
        if global_path is None:
            global_path = Path(__file__).resolve().parent / "settings.yaml"

        if not Path(global_path).exists():
            raise FileNotFoundError(f"Global settings file not found: {global_path}")

        with open(global_path, "r") as f:
            self.global_data = yaml.safe_load(f)

        # Handle .env loading from global settings
        dotenv_path = self.get("secrets", "dotenv_path")
        if dotenv_path:
            dotenv_file = Path(dotenv_path).expanduser()
            self._ensure_env_exists(dotenv_file)
            load_dotenv(dotenv_file)

    # Access helpers
    def get(self, *keys, default=None):
        """Retrieve nested keys from global settings.yaml."""
        node = self.global_data
        for key in keys:
            node = node.get(key, {}) if isinstance(node, dict) else None
        return node or default

    def resolve_api_key(self, api_key_name: str) -> str:
        """Fetch API key from environment."""
        if not api_key_name:
            return None
        value = os.getenv(api_key_name)
        if not value:
            if api_key_name == "OLLAMA_API_KEY":
                return "ollama"
            else:
                raise EnvironmentError(f"Missing environment variable: {api_key_name}")
        return value

    # Agent config loading
    def load_agent_config(self, agent_name: str):
        """
        Load agent-specific configuration from core/config/agents/<agent_name>.yaml
        """
        base_dir = Path(__file__).resolve().parent / "agents"
        agent_path = base_dir / f"{agent_name}.yaml"

        if not agent_path.exists():
            raise FileNotFoundError(f"Agent config not found at: {agent_path}")

        with open(agent_path, "r") as f:
            agent_conf = yaml.safe_load(f)

        return agent_conf

    # Internal helpers
    def _ensure_env_exists(self, dotenv_file: Path):
        """Ensure ~/.agenticai/.env exists (create it if missing)."""
        if not dotenv_file.parent.exists():
            dotenv_file.parent.mkdir(parents=True, exist_ok=True)

        if not dotenv_file.exists():
            dotenv_file.write_text(
                "# Environment variables for AgenticAI\n"
                "# Add your API keys here, for example:\n"
                "# FIREWORKS_API_KEY=your_api_key_here\n"
                "# OLLAMA_API_KEY=ollama\n"
            )
            print(f"[Settings] Created new .env file at {dotenv_file}")
