# AgenticCodingPrototype
Sample Agentic Coding Pipeline with RAG system.

## Ollama Setup (Apple silicon):
```bash
# Install Ollama
brew install ollama
# Start Ollama service (runs automatically afterward)
ollama serve
# Pull main model (instruct-tuned)
ollama pull llama3.1:8b-instruct-q4_K_M
# Pull embedding model for RAG
ollama pull nomic-embed-text
```

## Python Setup (Apple silicon):
```bash
# Install Python 3.9 (recommended for compatibility)
brew install python@3.9
# Create and activate a virtual environment
python3.9 -m venv venv
source ./venv/bin/activate
# Install dependencies
pip install -r requirements.txt
# Run the main app
python main.py
```

## How to use:
