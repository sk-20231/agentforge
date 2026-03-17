"""
Unit tests for agentforge.evaluation:
- load_eval_dataset (validation)
- validate_against_corpus (chunk ID cross-check)
- recall_at_k (pure metric function)
- score_faithfulness (mocked LLM judge)
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from agentforge.evaluation import (
    load_eval_dataset,
    validate_against_corpus,
    recall_at_k,
    score_faithfulness,
)


# ---------- recall_at_k (pure, no mocks) ----------

class TestRecallAtK:
    """Tests for recall_at_k — fraction of expected IDs found in retrieved list."""

    def test_perfect_recall(self):
        assert recall_at_k(["a", "b"], ["a", "b", "c"]) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["a", "b"], ["c", "d", "e"]) == 0.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "b", "c"], ["a", "c", "d"]) == pytest.approx(2 / 3)

    def test_single_expected_found(self):
        assert recall_at_k(["a"], ["a", "b"]) == 1.0

    def test_single_expected_missing(self):
        assert recall_at_k(["a"], ["b", "c"]) == 0.0

    def test_empty_expected_returns_one(self):
        assert recall_at_k([], ["a", "b"]) == 1.0

    def test_duplicates_in_retrieved_dont_inflate(self):
        assert recall_at_k(["a", "b"], ["a", "a", "a"]) == 0.5

    def test_order_does_not_matter(self):
        assert recall_at_k(["c", "a"], ["a", "b", "c"]) == 1.0


# ---------- load_eval_dataset (validation) ----------

class TestLoadEvalDataset:
    """Tests for load_eval_dataset — structure validation."""

    def _write_dataset(self, tmp_path, data):
        path = tmp_path / "eval.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    def _valid_entry(self, id="eval_001"):
        return {
            "id": id,
            "question": "What is X?",
            "expected_facts": ["X is Y"],
            "expected_chunk_ids": ["chunk_0"],
        }

    def test_loads_valid_dataset(self, tmp_path):
        path = self._write_dataset(tmp_path, [self._valid_entry()])
        result = load_eval_dataset(path)
        assert len(result) == 1
        assert result[0]["id"] == "eval_001"

    def test_rejects_empty_array(self, tmp_path):
        path = self._write_dataset(tmp_path, [])
        with pytest.raises(ValueError, match="non-empty"):
            load_eval_dataset(path)

    def test_rejects_non_array(self, tmp_path):
        path = self._write_dataset(tmp_path, {"id": "eval_001"})
        with pytest.raises(ValueError, match="non-empty"):
            load_eval_dataset(path)

    def test_rejects_missing_required_field(self, tmp_path):
        entry = self._valid_entry()
        del entry["question"]
        path = self._write_dataset(tmp_path, [entry])
        with pytest.raises(ValueError, match="missing fields"):
            load_eval_dataset(path)

    def test_rejects_duplicate_ids(self, tmp_path):
        path = self._write_dataset(tmp_path, [
            self._valid_entry("dup"),
            self._valid_entry("dup"),
        ])
        with pytest.raises(ValueError, match="Duplicate"):
            load_eval_dataset(path)

    def test_rejects_empty_expected_facts(self, tmp_path):
        entry = self._valid_entry()
        entry["expected_facts"] = []
        path = self._write_dataset(tmp_path, [entry])
        with pytest.raises(ValueError, match="expected_facts"):
            load_eval_dataset(path)

    def test_rejects_empty_expected_chunk_ids(self, tmp_path):
        entry = self._valid_entry()
        entry["expected_chunk_ids"] = []
        path = self._write_dataset(tmp_path, [entry])
        with pytest.raises(ValueError, match="expected_chunk_ids"):
            load_eval_dataset(path)

    def test_accepts_extra_fields(self, tmp_path):
        entry = self._valid_entry()
        entry["difficulty"] = "easy"
        entry["source"] = "test.txt"
        path = self._write_dataset(tmp_path, [entry])
        result = load_eval_dataset(path)
        assert result[0]["difficulty"] == "easy"

    def test_multiple_valid_entries(self, tmp_path):
        path = self._write_dataset(tmp_path, [
            self._valid_entry("e1"),
            self._valid_entry("e2"),
            self._valid_entry("e3"),
        ])
        result = load_eval_dataset(path)
        assert len(result) == 3


# ---------- validate_against_corpus ----------

class TestValidateAgainstCorpus:
    """Tests for validate_against_corpus — chunk ID cross-check."""

    def _write_corpus(self, tmp_path, chunk_ids):
        path = tmp_path / "corpus.json"
        corpus = [{"id": cid, "text": "t", "embedding": [], "source": "s"} for cid in chunk_ids]
        path.write_text(json.dumps(corpus), encoding="utf-8")
        return str(path)

    def test_all_ids_present(self, tmp_path):
        corpus_path = self._write_corpus(tmp_path, ["chunk_0", "chunk_1", "chunk_2"])
        dataset = [
            {"id": "e1", "expected_chunk_ids": ["chunk_0", "chunk_1"]},
        ]
        result = validate_against_corpus(dataset, corpus_path)
        assert result["valid"] is True
        assert result["missing"] == []

    def test_missing_ids_reported(self, tmp_path):
        corpus_path = self._write_corpus(tmp_path, ["chunk_0"])
        dataset = [
            {"id": "e1", "expected_chunk_ids": ["chunk_0", "chunk_99"]},
        ]
        result = validate_against_corpus(dataset, corpus_path)
        assert result["valid"] is False
        assert len(result["missing"]) == 1
        assert result["missing"][0]["chunk_id"] == "chunk_99"
        assert result["missing"][0]["eval_id"] == "e1"

    def test_multiple_entries_with_missing(self, tmp_path):
        corpus_path = self._write_corpus(tmp_path, ["a", "b"])
        dataset = [
            {"id": "e1", "expected_chunk_ids": ["a", "c"]},
            {"id": "e2", "expected_chunk_ids": ["b", "d"]},
        ]
        result = validate_against_corpus(dataset, corpus_path)
        assert result["valid"] is False
        assert len(result["missing"]) == 2


# ---------- score_faithfulness (mocked LLM) ----------

class TestScoreFaithfulness:
    """Tests for score_faithfulness — LLM-as-judge with mocked API."""

    CHUNKS = [
        {"id": "c0", "text": "Dogs are loyal pets."},
        {"id": "c1", "text": "Cats are independent."},
    ]

    @patch("openai.OpenAI")
    def test_faithful_verdict(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content='{"faithful": true, "reason": "All facts supported."}'))
        ]

        result = score_faithfulness("What are dogs?", "Dogs are loyal pets.", self.CHUNKS)
        assert result["faithful"] is True
        assert "supported" in result["reason"].lower()

    @patch("openai.OpenAI")
    def test_unfaithful_verdict(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content='{"faithful": false, "reason": "Answer invented a fact."}'))
        ]

        result = score_faithfulness("What are dogs?", "Dogs fly.", self.CHUNKS)
        assert result["faithful"] is False

    @patch("openai.OpenAI")
    def test_handles_api_error_gracefully(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API down")

        result = score_faithfulness("Q", "A", self.CHUNKS)
        assert result["faithful"] is False
        assert "Judge error" in result["reason"]

    @patch("openai.OpenAI")
    def test_handles_malformed_json_gracefully(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="not json"))
        ]

        result = score_faithfulness("Q", "A", self.CHUNKS)
        assert result["faithful"] is False
        assert "Judge error" in result["reason"]
