"""
Unit tests for agentforge.rag.document_store:
- chunk_text (pure function, no API)
- search_docs (mocked corpus + embedding)
- load_corpus / save_corpus (temp file)
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from agentforge.rag.document_store import chunk_text, search_docs, load_corpus, save_corpus, list_sources


# ---------- chunk_text (pure, no mocks) ----------

class TestChunkText:
    """Tests for chunk_text — paragraph-aware splitting with overlap."""

    def test_empty_input_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []
        assert chunk_text(None) == []

    def test_single_short_paragraph(self):
        result = chunk_text("Hello world.")
        assert result == ["Hello world."]

    def test_multiple_paragraphs_each_becomes_chunk(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = chunk_text(text, max_chars=500)
        assert len(result) == 3
        assert result[0] == "First paragraph."
        assert result[1] == "Second paragraph."
        assert result[2] == "Third paragraph."

    def test_long_paragraph_splits_with_overlap(self):
        long_para = "A" * 200
        result = chunk_text(long_para, max_chars=100, overlap=20)
        assert len(result) >= 2
        # First chunk is 100 chars
        assert len(result[0]) == 100
        # Overlap: second chunk starts 80 chars in (100 - 20)
        assert result[1][:20] == result[0][80:100]

    def test_short_paragraphs_not_split(self):
        text = "Short one.\n\nAnother short one."
        result = chunk_text(text, max_chars=500)
        assert len(result) == 2
        assert "Short one." in result[0]

    def test_overlap_does_not_cause_infinite_loop(self):
        """If overlap >= max_chars, the code falls back to no overlap."""
        result = chunk_text("A" * 50, max_chars=10, overlap=10)
        assert len(result) >= 1
        for c in result:
            assert len(c) <= 10


# ---------- load_corpus / save_corpus ----------

class TestCorpusPersistence:
    """Tests for load_corpus and save_corpus using a temp file."""

    def test_load_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "agentforge.rag.document_store.AGENT_CORPUS_FILE",
            str(tmp_path / "nonexistent.json"),
        )
        assert load_corpus() == []

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        path = str(tmp_path / "corpus.json")
        monkeypatch.setattr("agentforge.rag.document_store.AGENT_CORPUS_FILE", path)

        corpus = [
            {"id": "c0", "text": "hello", "embedding": [0.1, 0.2], "source": "test.txt"}
        ]
        save_corpus(corpus)
        loaded = load_corpus()
        assert len(loaded) == 1
        assert loaded[0]["id"] == "c0"
        assert loaded[0]["embedding"] == [0.1, 0.2]

    def test_load_invalid_json_returns_empty(self, tmp_path, monkeypatch):
        path = tmp_path / "bad.json"
        path.write_text("not json at all")
        monkeypatch.setattr(
            "agentforge.rag.document_store.AGENT_CORPUS_FILE", str(path)
        )
        assert load_corpus() == []

    def test_model_mismatch_raises_runtime_error(self, tmp_path, monkeypatch):
        """Corpus built with a different model → RuntimeError (fail-fast).
        This is the core guard against silent embedding space mismatch.
        """
        path = str(tmp_path / "corpus.json")
        monkeypatch.setattr("agentforge.rag.document_store.AGENT_CORPUS_FILE", path)

        # Write a corpus that claims a different embedding model
        data = {
            "embedding_model": "some-other-model-that-is-not-configured",
            "chunks": [{"id": "c0", "text": "hello", "embedding": [0.1], "source": "x.txt"}],
        }
        with open(path, "w") as f:
            json.dump(data, f)

        with pytest.raises(RuntimeError, match="(?i)embedding model mismatch"):
            load_corpus()

    def test_old_format_bare_list_loads_without_error(self, tmp_path, monkeypatch):
        """Corpus in old bare-list format (no model metadata) loads as-is.
        Backward compatibility: existing corpora created before this change
        still work — they just don't get model validation.
        """
        path = str(tmp_path / "corpus.json")
        monkeypatch.setattr("agentforge.rag.document_store.AGENT_CORPUS_FILE", path)

        old_corpus = [{"id": "c0", "text": "hello", "embedding": [0.1, 0.2], "source": "x.txt"}]
        with open(path, "w") as f:
            json.dump(old_corpus, f)

        loaded = load_corpus()
        assert len(loaded) == 1
        assert loaded[0]["id"] == "c0"

    def test_list_sources_returns_unique_sorted_sources(self, tmp_path, monkeypatch):
        """list_sources() returns unique filenames sorted — the checklist before deleting corpus."""
        path = str(tmp_path / "corpus.json")
        monkeypatch.setattr("agentforge.rag.document_store.AGENT_CORPUS_FILE", path)

        corpus = [
            {"id": "a_0", "text": "x", "embedding": [0.1], "source": "zebra.txt"},
            {"id": "a_1", "text": "y", "embedding": [0.2], "source": "alpha.txt"},
            {"id": "a_2", "text": "z", "embedding": [0.3], "source": "zebra.txt"},  # duplicate source
        ]
        save_corpus(corpus)

        sources = list_sources()
        assert sources == ["alpha.txt", "zebra.txt"]   # sorted, deduplicated

    def test_list_sources_empty_corpus_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "agentforge.rag.document_store.AGENT_CORPUS_FILE",
            str(tmp_path / "nonexistent.json"),
        )
        assert list_sources() == []

    def test_save_stores_embedding_model_name(self, tmp_path, monkeypatch):
        """save_corpus writes the model name into the file alongside the chunks.
        This is what enables load_corpus to detect mismatches later.
        """
        path = str(tmp_path / "corpus.json")
        monkeypatch.setattr("agentforge.rag.document_store.AGENT_CORPUS_FILE", path)

        save_corpus([{"id": "c0", "text": "hi", "embedding": [0.1], "source": "x.txt"}])

        with open(path) as f:
            raw = json.load(f)

        assert "embedding_model" in raw
        assert "chunks" in raw
        assert len(raw["chunks"]) == 1


# ---------- search_docs (mocked embedding + corpus) ----------

class TestSearchDocs:
    """Tests for search_docs with a tiny fixed corpus and mocked embedding."""

    TINY_CORPUS = [
        {"id": "dogs_0", "text": "Dogs are loyal pets.", "embedding": [1.0, 0.0, 0.0], "source": "animals.txt"},
        {"id": "cats_0", "text": "Cats are independent.", "embedding": [0.0, 1.0, 0.0], "source": "animals.txt"},
        {"id": "fish_0", "text": "Fish live in water.", "embedding": [0.0, 0.0, 1.0], "source": "animals.txt"},
    ]

    @patch("agentforge.rag.document_store.load_corpus")
    @patch("agentforge.rag.document_store.get_embedding")
    def test_returns_most_similar_chunk(self, mock_emb, mock_corpus):
        mock_corpus.return_value = self.TINY_CORPUS
        # Query embedding points toward dogs_0 ([1, 0, 0])
        mock_emb.return_value = [1.0, 0.0, 0.0]

        results = search_docs("Tell me about dogs", top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "dogs_0"
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-6)

    @patch("agentforge.rag.document_store.load_corpus")
    @patch("agentforge.rag.document_store.get_embedding")
    def test_returns_ranked_results(self, mock_emb, mock_corpus):
        mock_corpus.return_value = self.TINY_CORPUS
        # Query embedding is between dogs and cats
        mock_emb.return_value = [0.9, 0.1, 0.0]

        results = search_docs("pets", top_k=3)
        assert len(results) == 3
        # dogs_0 should be first (closer to [1,0,0])
        assert results[0]["id"] == "dogs_0"
        # fish_0 should be last (orthogonal)
        assert results[2]["id"] == "fish_0"

    @patch("agentforge.rag.document_store.load_corpus")
    @patch("agentforge.rag.document_store.get_embedding")
    def test_top_k_limits_results(self, mock_emb, mock_corpus):
        mock_corpus.return_value = self.TINY_CORPUS
        mock_emb.return_value = [0.5, 0.5, 0.5]

        results = search_docs("anything", top_k=2)
        assert len(results) == 2

    @patch("agentforge.rag.document_store.load_corpus")
    @patch("agentforge.rag.document_store.get_embedding")
    def test_result_shape(self, mock_emb, mock_corpus):
        mock_corpus.return_value = self.TINY_CORPUS
        mock_emb.return_value = [1.0, 0.0, 0.0]

        results = search_docs("dogs", top_k=1)
        r = results[0]
        assert "id" in r
        assert "text" in r
        assert "source" in r
        assert "score" in r
        # No embedding vector in the result
        assert "embedding" not in r

    @patch("agentforge.rag.document_store.load_corpus")
    def test_empty_corpus_returns_empty(self, mock_corpus):
        mock_corpus.return_value = []
        assert search_docs("anything") == []

    def test_blank_query_returns_empty(self):
        assert search_docs("") == []
        assert search_docs("   ") == []

    @patch("agentforge.rag.document_store.load_corpus")
    @patch("agentforge.rag.document_store.get_embedding")
    def test_skips_items_without_embedding(self, mock_emb, mock_corpus):
        mock_corpus.return_value = [
            {"id": "no_emb", "text": "No vector.", "embedding": [], "source": "x.txt"},
            {"id": "has_emb", "text": "Has vector.", "embedding": [1.0, 0.0], "source": "x.txt"},
        ]
        mock_emb.return_value = [1.0, 0.0]

        results = search_docs("test", top_k=5)
        assert len(results) == 1
        assert results[0]["id"] == "has_emb"
