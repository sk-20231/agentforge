"""
Unit tests for agentforge.rag.qa:
- _strip_invalid_citations (pure function, no mocks)
- _build_prompt (pure function)
- answer_from_docs (mocked search_docs + LLM client)
"""
import pytest
from unittest.mock import patch, MagicMock

from agentforge.rag.qa import _strip_invalid_citations, _build_prompt, answer_from_docs


# ---------- _strip_invalid_citations (pure, no mocks) ----------

class TestStripInvalidCitations:
    """Tests for the citation guardrail — pure text processing."""

    def test_keeps_valid_citations(self):
        answer = "Dogs are loyal [dogs_0] and friendly [dogs_1]."
        valid = {"dogs_0", "dogs_1"}
        assert _strip_invalid_citations(answer, valid) == answer

    def test_removes_invalid_citations(self):
        answer = "Dogs are loyal [dogs_0] and cats are nice [cats_0]."
        valid = {"dogs_0"}
        result = _strip_invalid_citations(answer, valid)
        assert "[dogs_0]" in result
        assert "[cats_0]" not in result

    def test_removes_all_when_none_valid(self):
        answer = "Info from [fake_0] and [fake_1]."
        valid = set()
        result = _strip_invalid_citations(answer, valid)
        assert "[" not in result

    def test_no_citations_unchanged(self):
        answer = "Just a plain answer with no citations."
        result = _strip_invalid_citations(answer, {"any_id"})
        assert result == answer

    def test_cleans_up_double_spaces(self):
        answer = "Start [bad_id] end."
        result = _strip_invalid_citations(answer, set())
        assert "  " not in result

    def test_mixed_valid_and_invalid(self):
        answer = "A [valid_0] B [invalid_0] C [valid_1] D [invalid_1]."
        valid = {"valid_0", "valid_1"}
        result = _strip_invalid_citations(answer, valid)
        assert "[valid_0]" in result
        assert "[valid_1]" in result
        assert "[invalid_0]" not in result
        assert "[invalid_1]" not in result


# ---------- _build_prompt ----------

class TestBuildPrompt:
    """Tests for prompt construction."""

    def test_includes_chunk_ids_and_text(self):
        chunks = [
            {"id": "c0", "text": "Dogs are great.", "source": "pets.txt", "score": 0.9},
            {"id": "c1", "text": "Cats are ok.", "source": "pets.txt", "score": 0.8},
        ]
        prompt = _build_prompt("Tell me about pets", chunks)
        assert "[c0]" in prompt
        assert "[c1]" in prompt
        assert "Dogs are great." in prompt
        assert "Cats are ok." in prompt
        assert "pets.txt" in prompt
        assert "Tell me about pets" in prompt

    def test_includes_rules(self):
        chunks = [{"id": "c0", "text": "Text.", "source": "s.txt", "score": 0.5}]
        prompt = _build_prompt("q", chunks)
        assert "ONLY" in prompt
        assert "cite" in prompt.lower() or "Cite" in prompt


# ---------- answer_from_docs (mocked search + LLM) ----------

class TestAnswerFromDocs:
    """Tests for the full RAG pipeline with mocked dependencies."""

    @patch("agentforge.rag.qa.log_event")
    @patch("agentforge.rag.qa._client")
    @patch("agentforge.rag.qa.search_docs")
    def test_returns_answer_with_valid_citations(self, mock_search, mock_client, mock_log):
        mock_search.return_value = [
            {"id": "doc_chunk_0", "text": "RAG uses retrieval.", "source": "doc.txt", "score": 0.95},
            {"id": "doc_chunk_1", "text": "Embeddings enable search.", "source": "doc.txt", "score": 0.90},
        ]

        # LLM returns answer citing both valid chunks
        msg = MagicMock()
        msg.content = "RAG uses retrieval [doc_chunk_0] and embeddings [doc_chunk_1]."
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        mock_client.chat.completions.create.return_value = response

        result = answer_from_docs("What is RAG?")
        assert "[doc_chunk_0]" in result
        assert "[doc_chunk_1]" in result

    @patch("agentforge.rag.qa.log_event")
    @patch("agentforge.rag.qa._client")
    @patch("agentforge.rag.qa.search_docs")
    def test_guardrail_strips_hallucinated_citation(self, mock_search, mock_client, mock_log):
        mock_search.return_value = [
            {"id": "real_0", "text": "Real chunk.", "source": "doc.txt", "score": 0.9},
        ]

        # LLM hallucinates a citation that wasn't in the retrieved set
        msg = MagicMock()
        msg.content = "Answer from [real_0] and also [hallucinated_99]."
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        mock_client.chat.completions.create.return_value = response

        result = answer_from_docs("question")
        assert "[real_0]" in result
        assert "[hallucinated_99]" not in result

    @pytest.mark.parametrize("end_trace, expected", [(True, 1), (False, 0)])
    @patch("agentforge.rag.qa.log_event")
    @patch("agentforge.rag.qa._client")
    @patch("agentforge.rag.qa.search_docs")
    def test_end_trace_flag_controls_the_trace_end_event(self, mock_search, mock_client,
                                                         mock_log, end_trace, expected):
        """Step 21.2: the supervisor's documents worker is one part of a turn,
        so it must not end the turn's trace (end_trace=False)."""
        mock_search.return_value = [
            {"id": "c0", "text": "Chunk.", "source": "doc.txt", "score": 0.9},
        ]
        msg = MagicMock()
        msg.content = "Answer [c0]."
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=msg)])

        answer_from_docs("question", end_trace=end_trace)
        trace_ends = [c for c in mock_log.call_args_list if c.args[0] == "trace_end"]
        assert len(trace_ends) == expected

    @patch("agentforge.rag.qa.search_docs")
    def test_no_docs_returns_fallback_message(self, mock_search):
        mock_search.return_value = []
        result = answer_from_docs("anything")
        assert "don't have any documents" in result.lower() or "ingest" in result.lower()

    @patch("agentforge.logger.log_event")
    @patch("agentforge.rag.qa._client")
    @patch("agentforge.rag.qa.search_docs")
    def test_logs_retrieval_and_answer(self, mock_search, mock_client, mock_log):
        mock_search.return_value = [
            {"id": "c0", "text": "Text.", "source": "s.txt", "score": 0.5},
        ]
        msg = MagicMock()
        msg.content = "Answer [c0]."
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        mock_client.chat.completions.create.return_value = response

        answer_from_docs("q")

        log_calls = [call[0][0] for call in mock_log.call_args_list]
        assert "docs_qa_retrieve" in log_calls
        assert "docs_qa_generate" in log_calls


# ---------- Conversation history tests ----------

class TestAnswerFromDocsWithHistory:
    """Tests that conversation history flows into the LLM call for RAG."""

    @patch("agentforge.rag.qa.log_event")
    @patch("agentforge.rag.qa._client")
    @patch("agentforge.rag.qa.search_docs")
    def test_history_included_in_llm_messages(self, mock_search, mock_client, mock_log):
        mock_search.return_value = [
            {"id": "c0", "text": "RAG stands for retrieval-augmented generation.", "source": "doc.txt", "score": 0.9},
        ]
        msg = MagicMock()
        msg.content = "It combines retrieval with generation [c0]."
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        mock_client.chat.completions.create.return_value = resp

        history = [
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": "RAG is retrieval-augmented generation."},
        ]
        answer_from_docs("How does it work?", history=history)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        contents = [m["content"] for m in messages]
        assert any("What is RAG?" in c for c in contents)
        assert messages[-1]["content"] == "How does it work?"

    @patch("agentforge.rag.qa.log_event")
    @patch("agentforge.rag.qa._client")
    @patch("agentforge.rag.qa.search_docs")
    def test_no_history_backward_compatible(self, mock_search, mock_client, mock_log):
        mock_search.return_value = [
            {"id": "c0", "text": "Text.", "source": "s.txt", "score": 0.5},
        ]
        msg = MagicMock()
        msg.content = "Answer [c0]."
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        mock_client.chat.completions.create.return_value = resp

        answer_from_docs("question")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 2  # system (RAG prompt) + user
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"


# ---------- streamed replies: the citation guardrail runs BEFORE text is shown (#79) ----------

from agentforge.rag.qa import (  # noqa: E402
    CITATION_HOLD_MAX_CHARS, _filter_citations, _normalize_spacing)

_VALID = {"Chunk_1", "doc_0", "["}

_REPLIES = [
    "Leave is 20 days [Chunk_1] and [Chunk_99].",       # the reproduction in #79
    "See [Chunk_99] for details [doc_0].",               # removal between two spaces
    "[Chunk_99] starts the reply.",                      # citation at the very start
    "Ends with a citation [Chunk_99]",                   # ... and at the very end
    "Ends with a valid one [Chunk_1]",
    "Empty [] brackets stay, [Chunk_1] too.",
    "Nested [[Chunk_1] and [[x] and [a [b] c.",
    "A lone [ with no closing bracket at all.",
    "Two  spaces,   many spaces, and\nnew lines  [Chunk_99]  here.",
    "  \n Leading and trailing whitespace [Chunk_99] \n ",
    "Back to back [Chunk_1][Chunk_99][doc_0].",
    "Close then open ][ and [Chunk_99",
    "",
    "   ",
    "No citations at all.",
]


def _pieces(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)]


def _streamed(text, size, valid=_VALID):
    return "".join(_normalize_spacing(_filter_citations(_pieces(text, size), valid)))


class TestStreamedCitationGuard:

    @pytest.mark.parametrize("size", [1, 3, 7, 1000])
    @pytest.mark.parametrize("reply", _REPLIES)
    def test_stream_equals_the_non_streamed_answer(self, reply, size):
        assert _streamed(reply, size) == _strip_invalid_citations(reply, _VALID)

    def test_random_replies_match_at_every_token_size(self):
        import random
        rng = random.Random(79)
        alphabet = ["a", "b", " ", " ", "\n", "[", "]", "Chunk_1", "doc_0", "x"]
        for _ in range(500):
            reply = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
            for size in (1, 2, 3, 5, 1000):
                assert _streamed(reply, size) == _strip_invalid_citations(reply, _VALID), \
                    (reply, size)

    def test_text_without_brackets_is_not_held_back(self):
        out = list(_filter_citations(["Hello ", "world"], set()))
        assert out == ["Hello ", "world"]

    def test_a_lone_bracket_does_not_hold_back_the_rest_of_the_reply(self):
        consumed = []

        def pieces():
            for i, ch in enumerate("Start [" + "x" * 1000):
                consumed.append(i)
                yield ch

        stream = _filter_citations(pieces(), set())
        next(stream)                           # "S"
        first_held = None
        for out in stream:
            if "[" in out:
                first_held = len(consumed)
                break
        # The "[" text is released once the hold limit is passed, not at the end.
        assert first_held is not None
        assert first_held <= len("Start [") + CITATION_HOLD_MAX_CHARS + 1


class TestStreamedAnswerFromDocs:

    @staticmethod
    def _stream(mock_client, tokens):
        chunks = []
        for t in tokens:
            c = MagicMock(usage=None)
            c.choices = [MagicMock(delta=MagicMock(content=t))]
            chunks.append(c)
        mock_client.chat.completions.create.return_value = iter(chunks)

    @pytest.mark.parametrize("size", [1, 3, 7, 1000])
    @patch("agentforge.rag.qa.log_event")
    @patch("agentforge.rag.qa._client")
    @patch("agentforge.rag.qa.search_docs")
    def test_invented_citation_is_never_shown(self, mock_search, mock_client, mock_log, size):
        mock_search.return_value = [
            {"id": "Chunk_1", "text": "Leave policy.", "source": "doc.txt", "score": 0.9}]
        raw = "Leave is 20 days [Chunk_1] and [Chunk_99]."
        self._stream(mock_client, _pieces(raw, size))

        pieces = list(answer_from_docs("How much leave?", stream=True))

        assert all("Chunk_99" not in p for p in pieces)
        assert "".join(pieces) == "Leave is 20 days [Chunk_1] and ."
        (answer,) = [c.args[1] for c in mock_log.call_args_list
                     if c.args[0] == "docs_qa_answer"]
        assert answer["citations_removed"] is True
        assert answer["stream_matches_clean"] is True
        assert answer["raw_answer_length"] == len(raw)
