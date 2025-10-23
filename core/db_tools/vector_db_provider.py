from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
)
from core.config.settings_loader import Settings
from core.embedding_tools.embedding_provider import EmbeddingProvider
import os
import shutil

class DummyLoader:
    def lazy_load(self):
        return []
    
class VectorDBProvider:
    """
    """

    def __init__(self, settings: Settings, agent_name: str, embedding_provider: EmbeddingProvider):
        self.agent_conf = settings.load_agent_config(agent_name)
        self.root_dir = self.agent_conf["project_root"]
        
        mem_conf = self.agent_conf["memory"]["vector_db"]

        self.mem_conf = mem_conf
        self.persist_dir = mem_conf["persist_directory"]
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)
        self.chunk_size = mem_conf["chunk_size"]
        self.chunk_overlap = mem_conf["chunk_overlap"]
        self.retriever_k = mem_conf.get("retriever_k", 3)

        # Initialize embeddings
        self.embeddings = embedding_provider.get_provider()

    # Document loading
    def is_binary(self, path):
        with open(path, 'rb') as f:
            chunk = f.read(4096)
        try:
            chunk.decode('utf-8')
            return False   # decodes fine -> text
        except UnicodeDecodeError:
            return True    # invalid UTF-8 -> probably binary

    def custom_loader(self, path):
        """
        Loads a file based on its extension, skipping binary files.
        """
        # print(path)

        ext = os.path.splitext(path)[1].lower()

        if self.is_binary(path):
            print(path, " is binary")
            return DummyLoader()  # Skip binary files
        else:
            print(path, " is not binary")

        if ext == ".pdf":
            return PyPDFLoader(path)
        elif ext in [".docx", ".doc"]:
            return UnstructuredWordDocumentLoader(path)
        else:
            return TextLoader(path, encoding="utf-8")

    def load_documents(self, source_dir: str):
        """Loads text and document files from a given directory."""
        source = source_dir
        loader = DirectoryLoader(
            source,
            glob="**/*",
            loader_cls=self.custom_loader,
            show_progress=True,
        )
        docs = loader.load()
        print(f"[VectorDBManager] Loaded {len(docs)} documents.")
        return docs

    # Build or load
    def build(self, source_dir: str):
        """
        Rebuilds a fresh Chroma DB from documents.
        Automatically deletes any existing DB directory before rebuild.
        """
        if self.persist_dir.exists():
            print(f"[VectorDBManager] Purging existing DB at {self.persist_dir}...")
            shutil.rmtree(self.persist_dir, ignore_errors=True)

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        docs = self.load_documents(source_dir)
        if not docs:
            print(f"[VectorDBManager] No documents found in {source_dir}")
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(docs)

        db = Chroma.from_documents(
            chunks,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir),
        )
        db.persist()

        print(f"[VectorDBManager] Rebuilt DB at {self.persist_dir}")
        return db

    def load_or_create(self):
        """Load existing DB, or create an empty one if missing."""
        try:
            db = Chroma(
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_dir),
            )
            retriever = db.as_retriever(search_kwargs={"k": self.retriever_k})
            print(f"[VectorDBManager] Loaded existing DB from {self.persist_dir}")
            return db, retriever
        except Exception:
            print(f"[VectorDBManager] No existing DB found at {self.persist_dir}, initializing empty.")
            db = Chroma.from_documents([], embedding=self.embeddings, persist_directory=str(self.persist_dir))
            retriever = db.as_retriever(search_kwargs={"k": self.retriever_k})
            return db, retriever
    
    # Single-file update
    def upsert_file(self, file_path: str):
        """
        Replace all previous entries for a file and insert the updated version.
        Safe to call multiple times — ensures no duplicate vectors.
        """
        db = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )

        # 1. Delete existing vectors by metadata match
        db.delete(where={"source": file_path})
        # print(f"[VectorDBManager] Cleared existing entries for {file_path}")

        # 2. Load new content
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        _loader = TextLoader(file_path, encoding="utf-8")
        _docs = _loader.load()
        _chunks = splitter.split_documents(_docs)
        db.add_documents(_chunks)

        # print(f"[VectorDBManager] Updated DB with {len(_chunks)} chunks from {file_path}")
