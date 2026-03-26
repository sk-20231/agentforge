# agent.memory — semantic memory, embedding, memory-aware responses
from agentforge.memory.semantic import (
    load_memory,
    save_memory,
    get_embedding,
    cosine_similarity,
    get_relevant_memories,
    store_memory,
)
from agentforge.memory.response import answer_with_memory
