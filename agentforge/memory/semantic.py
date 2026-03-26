# agent/memory/semantic.py
from openai import OpenAI
import json
import os
import numpy as np

from agentforge.config import AGENT_MEMORY_DIR, OPENAI_EMBEDDING_MODEL, OPENAI_BASE_URL
from agentforge.logger import log_event

client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()
MEMORY_DIR = AGENT_MEMORY_DIR
os.makedirs(MEMORY_DIR, exist_ok=True)


# ---------- Persistence ----------

def _get_user_memory_file(user_id: str):
    return os.path.join(MEMORY_DIR, f"user_{user_id}.json")

def load_memory(user_id: str):
    path = _get_user_memory_file(user_id)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        content = f.read().strip()
        return json.loads(content) if content else []



def save_memory(user_id: str, memory):
    path = _get_user_memory_file(user_id)
    with open(path, "w") as f:
        json.dump(memory, f, indent=2)


# ---------- Vector Math ----------

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------- Embeddings ----------

def get_embedding(text: str):
    # Validate input before calling API
    if not text or not isinstance(text, str) or not text.strip():
        raise ValueError(f"Invalid input for embedding: {repr(text)}")
    
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=text.strip()
    )
    return response.data[0].embedding


# ---------- Public Memory API ----------

def get_relevant_memories(user_id: str, query: str, top_k: int = 3):
    memory = load_memory(user_id)
    if not memory:
        return []
    
    # Validate query before embedding
    if not query or not isinstance(query, str) or not query.strip():
        return []

    query_embedding = get_embedding(query.strip())

    scored = [
        (cosine_similarity(query_embedding, item["embedding"]), item["text"])
        for item in memory
    ]

    scored.sort(reverse=True)
    return [text for _, text in scored[:top_k]]


def store_memory(user_id: str, text: str):
    # Validate inputs
    if not text or not isinstance(text, str) or not text.strip():
        print(f"Warning: Skipping invalid memory text: {repr(text)}")
        return
    
    text = text.strip()
    memory = load_memory(user_id)
    embedding = get_embedding(text)
    log_event("memory_write", {"text": text})

    memory.append({
        "text": text,
        "embedding": embedding
    })

    save_memory(user_id, memory)
