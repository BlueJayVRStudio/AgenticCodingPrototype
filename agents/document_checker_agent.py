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

class DocumentCheckerAgent:
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
        
        self.messages=[]
        self.instruct_message_base = [
            {"role": "system", "content": f"""
            You always responds in valid JSON blob.
            
            Your job is to perform lightweight document inspection:
            DO:
                - catch spelling and punctuation errors
                - catch sensitive information such as passwords and api keys
                - catch unexpected profanities
                - catch any other fatal mistakes
            DO NOT:
                - be too nitpicky

            Please return return your verdict as true or false for pass or fail, respectively.
            If pass return an empty string for suggested_edit. If fail, return the entire edited document for suggested_edit.
            
            Examples:
            
            **Example 1**:
             
            User Query:
            "We're going to a picnic!"
            
            Your Response:
            {{"verdict": true, "suggested_edit": ""}}
            
            **Example 2**:
            
            User Query:
            "We're going to a pcnic!"
             
            Your Response:
            {{"verdict": false, "suggested_edit": "We're going to a picnic!"}}
             
            **Example 3**:
            
            User Query:
            "Were going to a picnic!"
             
            Your Response:
            {{"verdict": false, "suggested_edit": "We're going to a picnic!"}}
             
            **Example 4**:
            
            User Query:
            "We're going to a picnic it'll be fun!"
             
            Your Response:
            {{"verdict": false, "suggested_edit": "We're going to a picnic. It'll be fun!"}}
            """},
        ]
    
    def generate_query(self, content):
        return {"role": "user", "content": content}
    
    def run(self, user_query):
        self.messages.append(self.generate_query(user_query))

        resp = self.llm_chat_completion_provider.chat_completion(self.instruct_message_base + self.messages)

        self.messages.pop()

        return resp.choices[0].message.content

    
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
