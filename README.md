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
### Following tools are available:

- get_weather — Retrieve the current weather for a city and country (mock implementation — prints a sample weather report)
- create_directory — Create a new folder within ./my_docs/
- save_code_output — Generate code or document and save it in ./my_docs/
- modify_document — Edit or update existing documents in ./my_docs/
- email_categorizer — Categorize messages as Spam, Urgent, Invite, or Promotion
- add_nums — Return the sum of two numbers
- query_rag — Query the Retrieval-Augmented Generation (RAG) system for contextual knowledge

You can ask any question in natural language, and the agent will automatically route it to the appropriate tool.
If it cannot recognize the intent, or if the user insists on casual conversation, it gracefully falls back to conversation mode.

### Sample Work Flow:
```text
Ask: can you help me decide if I need to bring a jacket to London next week?
The weather in London, UK is 22°C and sunny. This is from a test tool function! Great job!
TASK COMPLETED
Ask: can you filter this message for me: "we're having a buy one get one free sale!"
Promotion
TASK COMPLETED
Ask: can you write a short python script to count the frequency of words in a document and save it under word_frequency.py?
File word_frequency.py not found.
Modified and saved code output to ./my_docs/word_frequency.py
word_frequency.py
TASK COMPLETED
Ask: can you wrap it in a class?                          
Modified and saved code output to ./my_docs/word_frequency.py
word_frequency.py
TASK COMPLETED
Ask: Alright let's have a conversation. What have we been talking about so far?
We've been discussing ways to help you with various tasks. We started with getting the current weather for London, then filtered an email promotion message, generated a Python script to count word frequencies in a document, and finally wrapped that script in a class. How's your day going so far?
CONVERSATION COMPLETE
Ask: 
```

### Near-Term Goals

1. Optimize Short-Term Memory
    - Implement LLM-based pruning of short-term context to reduce token usage while keeping relevant information.
    - Integrate the existing Retrieval-Augmented Generation (RAG) system into semantic recall for improved contextual referencing.
2. Refactor Code Handling
    - The modify_document tool is stable and functions effectively for structured edits.
    - The save_code_output tool is less reliable and may be removed or restructured to decouple task routing from generative logic.