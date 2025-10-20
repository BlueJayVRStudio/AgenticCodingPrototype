import os
import json_repair
from datetime import datetime
from openai import OpenAI
from langchain.chains import RetrievalQA
from core.config.settings_loader import Settings
from core.utils.func_build_tools import build_tools_from_functions, get_args_in_order
from core.db_tools.vector_db_manager import VectorDBManager
from core.llm_tools.llm_manager import LLMManager

class BaseAgent:
    def __init__(self, agent_name: str = "base_agent"):
        self.agent_name = agent_name
        # Load settings
        self.settings = Settings()
        # Load agent-specific configuration
        self.agent_conf = self.settings.load_agent_config(agent_name)
        self.root_dir = self.agent_conf["project_root"]

        # LLMs
        self.llm_manager = LLMManager(self.settings, agent_name)
        self.llm = self.llm_manager.get_chat_llm()
        self.client = self.llm_manager.get_client()

        # Vector DB
        self.vector_manager = VectorDBManager(self.settings, agent_name)
        self.db, self.retriever, self.embeddings = self.vector_manager.load_or_create()

        # RetrievalQA chain
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
        )

        # Tools
        self.tools = [
            self.get_weather,
            self.create_directory,
            self.modify_document,
            self.email_categorizer,
            self.add_nums,
            self.query_rag,
        ]
        self.func_descriptions, self.func_lookup = build_tools_from_functions(self.tools)

        self.messages=[]
        self.instruct_message_base = [
            {"role": "system", "content": f"""
            You are a friendly and patient AI agent that always responds in valid JSON.
            
            You can use the following tools:
            {self.func_descriptions}
            Respond only in JSON like: {{"tool": "tool_name", "arguments": {{...}}}}, even
            if the user is asking for generative content. This is because your job is to
            route the request to proper tools. Please follow this protocol.

            However, if the user's intent is a conversation, simply return a plaintext response
            to the user in a friendly manner
            """},
        ]
        self.generative_message_base=[
            {"role": "system", "content": f"""
            You generate codes and documents. Respond only with the generated content.
            Do not keep and quotations or metadata around the content. Return solely
            the content. This is a strict requirement, please follow the protocol.
            """},
        ]

    def create_directory(self, relative_path):
        """
        Create target directory
        
        Arguments:
            relative_path (string): Relative path to the target directory.
        """

        dir_path = os.path.join(self.root_dir, relative_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
    def get_weather(self, city: str, country: str) -> str:
        """Return the current weather for a city and country."""
        return f"The weather in {city}, {country} is 22°C and sunny. This is from a test tool function! Great job!"

    def email_categorizer(self, category: str) -> str:
        """
        Return the content and category of an email
        
        Arguments:
            category (string): one of following categories
                Spam,
                Urgent,
                Invite,
                Promotion
        """
        return f"{category}" #: {content}"

    def add_nums(self, a: str, b: str) -> str:
        """
        Return the sum of a and b
        
        Arguments:
            a: string form of integer
            b: string form of integer
        """
        return "the sum between A and B is: " + str(int(a) + int(b))

    def query_rag(self, query: str) -> str:
        """
        Query the existing RAG (Retrieval-Augmented Generation) system for an answer.
        Use this tool when the user asks a question related to the local documents.
        """
        response = self.qa.invoke({"query": query})
        
        return f"""
        this is what I found out about your request: 
        {response["result"]}

        Supporting Documents:
        {response["source_documents"]}
        """

    def read_file(self, filename):
        try:
            file_path = os.path.join(self.root_dir, filename)
            with open(file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f'File {filename} not found.')
            return ""
        except Exception as e:
            print(f'An error occurred: {e}')
            return ""

    def modify_document(self, filename) -> str:
        """
        Modify a preexisting code or document. Inspect the content before generating a new suggestion.
        
        Arguments:
            filename (string): The file path of the content to modify (e.g., 'dir0/dir1/script.py').
        """
        content = self.read_file(filename)
        result = self.generate(f"""
        The following is the content to modify, please modify according the most recent request: 
                               
        {content}

        END OF CONTENT. Please ensure that if the content is a code that it is not wrapped in quotes 
        or any other extraneous texts that are not part of the script. Similarly detect extraneous
        details for other types of documents and exlude from the output.

        This is a final reminder that you should only output the content, not any of your commentaries.
        """) # this falls under 'user' query

        # print(f"modified content: {result}")
        print("Document Modification in Progress")
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output_{timestamp}.py"
        file_path = os.path.join(self.root_dir, filename)
        with open(file_path, "w") as f:
            f.write(result)
        
        self.vector_manager.upsert_file(file_path=file_path)

        print(f"Modified and saved code output to {file_path}")
    
        # self.messages.pop()
        # self.messages.pop()
        return filename
    
    def generate_assistant(self, content):
        return {"role": "assistant", "content": content}

    def generate_query(self, content):
        return {"role": "user", "content": content}

    def generate(self, user_query):
        self.messages.append(self.generate_query(user_query))

        resp = self.llm_manager.chat_completion(self.generative_message_base + self.messages)

        content = resp.choices[0].message.content
        self.messages.append(self.generate_assistant(content))
        return content
    
    def run(self, user_query):
        self.messages.append(self.generate_query(user_query))

        resp = self.llm_manager.chat_completion(self.instruct_message_base + self.messages)

        self.messages.append(self.generate_assistant(resp.choices[0].message.content))

        try:
            json_object = json_repair.loads(resp.choices[0].message.content)
            result = self.func_lookup[json_object["tool"]](*get_args_in_order(self.func_lookup[json_object["tool"]], json_object["arguments"]))
            return result, True
        except Exception as e:
            return resp.choices[0].message.content, False

