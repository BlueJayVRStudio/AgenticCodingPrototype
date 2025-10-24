import os
import json_repair
import json
from datetime import datetime
from openai import OpenAI
from langchain.chains import RetrievalQA
from core.utils.func_build_tools import build_tools_from_functions, get_args_in_order

from core.config.project_root_provider import ProjectRootProvider
from core.db_tools.vector_db_provider import VectorDBProvider
from core.llm_tools.llm_chat_provider import LLMChatProvider
from core.llm_tools.llm_chat_completion_provider import LLMChatCompletionProvider

class BaseAgent:
    def __init__(
            self, 
            project_root_provider: ProjectRootProvider,
            vector_db_provider: VectorDBProvider, 
            llm_chat_provider: LLMChatProvider, 
            llm_chat_completion_provider: LLMChatCompletionProvider, 
            agent_name: str
        ):
        self.agent_name = agent_name
        # Load agent-specific configuration
        self.root_dir = project_root_provider.root_dir

        # LLMs
        self.llm_chat_provider = llm_chat_provider
        self.llm_chat_completion_provider = llm_chat_completion_provider
        self.llm = self.llm_chat_provider.get_chat_llm()
        self.client = self.llm_chat_completion_provider.get_client()

        # Vector DB
        self.vector_db_provider = vector_db_provider
        self.db, self.retriever = self.vector_db_provider.load_or_create()

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
            self.create_document,
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
            
            Your sole job is to route requests to appropriate tools. Please return 
            required number of tool calls as a comma separated list of JSON maps:

            {{"tool": "tool_name", "arguments": {{...}}}},
            {{"tool": "tool_name", "arguments": {{...}}}},
            {{"tool": "tool_name", "arguments": {{...}}}},
            ...

            Remember, if the intent is to use a tool, only use tools from the given list
            of tools. Do not generate a false tool call.            
            
            However, if the user's intent is a conversation, simply return a plaintext response
            with the header "CONVERSATION: " in all caps to the user in a friendly manner. This is 
            the only exception to the routing rule. Example:

            CONVERSATION: This is a sample message!
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
        return f"The sum between {a} and {b} is: {int(a) + int(b)}"

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
        The following is the content to modify (from filename: {filename}), please modify according 
        to the most recent relevant requests: 
                               
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
        
        self.vector_db_provider.upsert_file(file_path=file_path)

        print(f"Modified and saved code output to {file_path}")
        return filename
    
    def create_document(self, filename) -> str:
        """
        Create new file at given file path.

        Arguments:
            filename (string): The path of the file to create (e.g., 'dir0/dir1/script.py').
        """
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output_{timestamp}.py"
        file_path = os.path.join(self.root_dir, filename)
        with open(file_path, "w") as f:
            f.write("")
        
        print(f"Created new file at {file_path}")
    
        return filename
    
    def generate_assistant(self, content):
        return {"role": "assistant", "content": content}

    def generate_query(self, content):
        return {"role": "user", "content": content}

    def generate(self, user_query):
        self.messages.append(self.generate_query(user_query))
        resp = self.llm_chat_completion_provider.chat_completion(self.generative_message_base + self.messages)

        content = resp.choices[0].message.content
        self.messages.append(self.generate_assistant(content))
        return content
    
    def run(self, user_query):
        self.messages.append(self.generate_query(user_query))

        resp = self.llm_chat_completion_provider.chat_completion(self.instruct_message_base + self.messages)

        self.messages.append(self.generate_assistant(resp.choices[0].message.content))

        header = "CONVERSATION:"

        if len(resp.choices[0].message.content) >= len(header) and resp.choices[0].message.content[:len(header)] == header:
            return [resp.choices[0].message.content[len(header):]], False
        else:
            tool_calls = self.extract_root_json_maps(resp.choices[0].message.content)
            # quit()
            # print(tool_calls)
            try:
                results = []
                for call in tool_calls:
                    json_object = json_repair.loads(call)
                    self.messages.append(self.generate_assistant(call))
                    result = self.func_lookup[json_object["tool"]](*get_args_in_order(self.func_lookup[json_object["tool"]], json_object["arguments"]))
                    results.append(result)
                return results, True
            except Exception as e:
                # print(f"Exception!: {e}")
                return [resp.choices[0].message.content], False
    
    def extract_root_json_maps(self, text: str):
        maps = []
        depth = 0
        start = None

        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    maps.append(text[start:i+1])
                    start = None

        return maps
