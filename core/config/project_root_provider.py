from core.config.settings_loader import Settings

class ProjectRootProvider:
    """
    Provides an agent's root project directory.
    """
    def __init__(self, settings: Settings, agent_name: str):
        self.agent_conf = settings.load_agent_config(agent_name)
        self.root_dir = self.agent_conf["project_root"]
    