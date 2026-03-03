"""
Configuration for the agent. Reads from environment variables with sensible defaults.
Optional: create a .env file or set env vars (OPENAI_API_KEY, OPENAI_MODEL, etc.).
"""
import os

# Optional: load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# OpenAI
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # None = use default

# Paths
AGENT_MEMORY_DIR = os.environ.get("AGENT_MEMORY_DIR", "memory")
AGENT_LOG_FILE = os.environ.get("AGENT_LOG_FILE", "agent_logs.jsonl")
# Single JSON file where the document corpus (chunks + embeddings) is stored for RAG.
AGENT_CORPUS_FILE = os.environ.get("AGENT_CORPUS_FILE", "corpus.json")

# Maximum estimated tokens allowed in conversation history before trimming.
# Conservative default leaves room for system prompts + RAG chunks (which can
# consume 1,000–2,000 tokens on DOCS_QA calls) within most model context windows.
HISTORY_TOKEN_BUDGET = int(os.environ.get("HISTORY_TOKEN_BUDGET", "2000"))
