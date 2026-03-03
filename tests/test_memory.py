"""
Unit tests for agent.memory.semantic: load_memory, save_memory, cosine_similarity,
get_relevant_memories (with mocked get_embedding), store_memory (with mocked get_embedding).
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from agentforge.memory.semantic import (
    load_memory,
    save_memory,
    cosine_similarity,
    get_relevant_memories,
    store_memory,
    _get_user_memory_file,
)


class TestCosineSimilarity:
    """Tests for cosine_similarity (pure math, no API)."""

    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-9

    def test_opposite_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-9


class TestLoadSaveMemory:
    """Tests for load_memory and save_memory with a temp directory."""

    def test_load_memory_empty_when_no_file(self, temp_memory_dir):
        assert load_memory("user_xyz") == []

    def test_save_and_load_memory_roundtrip(self, temp_memory_dir):
        user_id = "test_user_roundtrip"
        memory = [
            {"text": "I like dogs", "embedding": [0.1, 0.2, 0.3]},
        ]
        save_memory(user_id, memory)
        loaded = load_memory(user_id)
        assert len(loaded) == 1
        assert loaded[0]["text"] == "I like dogs"
        assert loaded[0]["embedding"] == [0.1, 0.2, 0.3]

    def test_get_user_memory_file(self, temp_memory_dir):
        path = _get_user_memory_file("alice")
        assert "alice" in path
        assert path.endswith(".json")


class TestGetRelevantMemories:
    """Tests for get_relevant_memories with mocked get_embedding."""

    def test_empty_memory_returns_empty_list(self, temp_memory_dir):
        with patch("agentforge.memory.semantic.get_embedding") as mock_emb:
            mock_emb.return_value = [0.1, 0.2]
            result = get_relevant_memories("user_empty", "some query")
        assert result == []

    def test_returns_top_k_by_similarity(self, temp_memory_dir):
        # Pre-populate memory file
        user_id = "user_with_memories"
        query_embedding = [1.0, 0.0, 0.0]
        memory = [
            {"text": "less relevant", "embedding": [0.0, 1.0, 0.0]},
            {"text": "most relevant", "embedding": [1.0, 0.0, 0.0]},
            {"text": "mid relevant", "embedding": [0.9, 0.1, 0.0]},
        ]
        save_memory(user_id, memory)

        with patch("agentforge.memory.semantic.get_embedding", return_value=query_embedding):
            result = get_relevant_memories(user_id, "query", top_k=2)
        assert len(result) == 2
        assert result[0] == "most relevant"
        assert result[1] == "mid relevant"

    def test_empty_query_returns_empty_list(self, temp_memory_dir):
        save_memory("u", [{"text": "x", "embedding": [0.1]}])
        result = get_relevant_memories("u", "   ")
        assert result == []


class TestStoreMemory:
    """Tests for store_memory with mocked get_embedding."""

    @patch("agentforge.memory.semantic.get_embedding")
    @patch("agentforge.memory.semantic.log_event")
    def test_store_memory_appends_and_persists(self, mock_log, mock_emb, temp_memory_dir):
        mock_emb.return_value = [0.5, 0.5]
        user_id = "store_test_user"
        store_memory(user_id, "I prefer tea")
        loaded = load_memory(user_id)
        assert len(loaded) == 1
        assert loaded[0]["text"] == "I prefer tea"
        assert loaded[0]["embedding"] == [0.5, 0.5]

    @patch("agentforge.memory.semantic.log_event")
    def test_store_memory_skips_empty_string(self, mock_log, temp_memory_dir):
        store_memory("u", "   ")
        assert load_memory("u") == []
