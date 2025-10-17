import os
from datetime import datetime
from openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from core.build_tools import *
import json_repair
import json

# from dotenv import load_dotenv
# load_dotenv("/Volumes/T9/ActualDesktop/Secrets/SECRETS.txt")
# FW_TOKEN = os.getenv("FW_TOKEN")

import shutil
shutil.rmtree("./vector_db", ignore_errors=True)
print("deleted vector db")

# Load documents
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
)

class SingleAgent():
    def __init__(self):
        self.root_dir = "./my_docs"
        os.makedirs(self.root_dir, exist_ok=True)
        self.loader = DirectoryLoader(
            self.root_dir,
            glob="**/*",
            loader_cls=self.custom_loader,
            show_progress=True
        )
        self.docs = self.loader.load()
        if not self.docs:
            file_path = os.path.join(self.root_dir, "dummy.txt")
            with open(file_path, "w") as f:
                f.write("hello world!")
            self.docs = self.loader.load()

        # Split into chunks
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.chunks = self.splitter.split_documents(self.docs)
        # Store with embeddings
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        # Initialize DB
        self.db = Chroma.from_documents(self.chunks, embedding=self.embeddings, persist_directory="./vector_db")
        self.llm = ChatOllama(model="llama3.1:8b-instruct-q4_K_M")
        self.retriever = self.db.as_retriever(search_kwargs={"k": 10})
        self.qa = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever, return_source_documents=True)
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"  # dummy key
        )
        # Initialize with Fireworks parameters
        # self.client = OpenAI(
        #     base_url="https://api.fireworks.ai/inference/v1",
        #     api_key=FW_TOKEN,
        # )

        self.tools = [self.get_weather, self.save_code_output, self.modify_document, self.email_categorizer, self.add_nums, self.query_rag]
        self.func_descriptions, self.func_lookup = build_tools_from_functions(self.tools)

        self.messages=[]
        self.instruct_message_base = [
            {"role": "system", "content": f"""
            You are a friendly and patient AI agent that always responds in valid JSON.
            
            You can use the following tools:
            {self.func_descriptions}
            Respond only in JSON like: {{"tool": "tool_name", "arguments": {{...}}}}.
            Finally, if the user insists on conversing, simply return a plaintext response to the user in a friendly manner
            """},
        ]
        self.generative_message_base=[
            {"role": "system", "content": f"""
            You generate codes and documents. Respond only with the generated content.
            Do not keep and quotations or metadata around the content. Return solely
            the content. This is a strict requirement, please follow the protocol.
            """},
        ]

    def custom_loader(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return PyPDFLoader(path)
        elif ext in [".docx", ".doc"]:
            return UnstructuredWordDocumentLoader(path)
        else:
            return TextLoader(path, encoding="utf-8")

    def save_code_output(self, code: str, filename: str) -> str:
        """
        Save generated code or document to a file path on disk.
        
        Arguments:
            code (string): The source code or document to save.
            filename (string): The file path to save the code under (e.g., 'dir0/dir1/script.py').
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output_{timestamp}.py"
        file_path = os.path.join(self.root_dir, filename)
        with open(file_path, "w") as f:
            f.write(code)
        
        _loader = TextLoader(file_path, encoding="utf-8")
        _docs = _loader.load()
        _chunks = self.splitter.split_documents(_docs)
        self.db.add_documents(_chunks)

        print(f"Saved code output to {file_path}")
        return file_path

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
        result = self.generate(f"This is the content to modify, please only return its modified version: {content}")
        # print(f"modified content: {result}")

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output_{timestamp}.py"
        file_path = os.path.join(self.root_dir, filename)
        with open(file_path, "w") as f:
            f.write(result)
        
        _loader = TextLoader(file_path, encoding="utf-8")
        _docs = _loader.load()
        _chunks = self.splitter.split_documents(_docs)
        self.db.add_documents(_chunks)

        print(f"Modified and saved code output to {file_path}")
    
        # self.messages.pop()
        # self.messages.pop()
        return filename
    
    def generate_assistant(self, content):
        return json.dumps({"role": "assistant", "content": content})

    def generate_query(self, content):
        return json.dumps({"role": "user", "content": content})

    def generate(self, user_query):
        self.messages.append(json.loads(self.generate_query(user_query)))

        resp = self.client.chat.completions.create(
            # model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            model="llama3.1:8b-instruct-q4_K_M",
            # model="qwen3:8b",
            messages=self.generative_message_base + self.messages,
            temperature=0.0,
            max_tokens=1024,
        )

        content = resp.choices[0].message.content
        self.messages.append(json_repair.loads(self.generate_assistant(content)))
        return content
    
    def run(self, user_query):
        self.messages.append(json.loads(self.generate_query(user_query)))

        resp = self.client.chat.completions.create(
            # model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            model="llama3.1:8b-instruct-q4_K_M",
            # model="qwen3:8b",
            messages=self.instruct_message_base + self.messages,
            temperature=0.0,
            max_tokens=1024,
        )

        self.messages.append(json_repair.loads(self.generate_assistant(resp.choices[0].message.content)))
        # print(self.messages)

        try:
            json_object = json_repair.loads(resp.choices[0].message.content)
            result = self.func_lookup[json_object["tool"]](*get_args_in_order(self.func_lookup[json_object["tool"]], json_object["arguments"]))
            return result, True
        except Exception as e:
            return resp.choices[0].message.content, False
        

