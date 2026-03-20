"""
Unit tests for the structured trace logger (Step 13).
Covers: generate_trace_id, log_event with trace fields, Span, compute_latency_percentiles.
"""
import json
import os
import time
import tempfile

import pytest
from unittest.mock import patch

from agentforge.logger import generate_trace_id, log_event, Span, compute_latency_percentiles


class TestGenerateTraceId:

    def test_returns_12_char_hex(self):
        tid = generate_trace_id()
        assert len(tid) == 12
        int(tid, 16)  # raises if not hex

    def test_unique_on_successive_calls(self):
        ids = {generate_trace_id() for _ in range(100)}
        assert len(ids) == 100


class TestLogEvent:

    def test_basic_event_written(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        with patch("agentforge.logger.AGENT_LOG_FILE", log_file):
            log_event("test_event", {"key": "value"})
        with open(log_file, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert record["event"] == "test_event"
        assert record["payload"]["key"] == "value"
        assert "timestamp" in record

    def test_trace_id_included_when_provided(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        with patch("agentforge.logger.AGENT_LOG_FILE", log_file):
            log_event("evt", {"x": 1}, trace_id="abc123def456")
        with open(log_file, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert record["trace_id"] == "abc123def456"

    def test_trace_id_absent_when_not_provided(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        with patch("agentforge.logger.AGENT_LOG_FILE", log_file):
            log_event("evt", {"x": 1})
        with open(log_file, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert "trace_id" not in record

    def test_duration_included_when_provided(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        with patch("agentforge.logger.AGENT_LOG_FILE", log_file):
            log_event("evt", {}, duration_ms=123.456)
        with open(log_file, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert record["duration_ms"] == 123.5

    def test_duration_absent_when_not_provided(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        with patch("agentforge.logger.AGENT_LOG_FILE", log_file):
            log_event("evt", {})
        with open(log_file, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert "duration_ms" not in record


class TestSpan:

    def test_logs_event_with_duration(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        with patch("agentforge.logger.AGENT_LOG_FILE", log_file):
            with Span("test_span", trace_id="tid123") as s:
                s.payload = {"detail": "yes"}
                time.sleep(0.05)

        with open(log_file, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert record["event"] == "test_span"
        assert record["trace_id"] == "tid123"
        assert record["duration_ms"] >= 40
        assert record["payload"]["detail"] == "yes"

    def test_records_error_on_exception(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        with patch("agentforge.logger.AGENT_LOG_FILE", log_file):
            with pytest.raises(ValueError):
                with Span("failing_span") as s:
                    raise ValueError("boom")

        with open(log_file, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert record["payload"]["error"] == "boom"
        assert record["duration_ms"] >= 0

    def test_span_without_trace_id(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        with patch("agentforge.logger.AGENT_LOG_FILE", log_file):
            with Span("no_tid_span"):
                pass
        with open(log_file, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert "trace_id" not in record


class TestComputeLatencyPercentiles:

    def _write_log(self, path, records):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_computes_p50_and_p95(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        records = [
            {"event": "step_a", "duration_ms": i * 10}
            for i in range(1, 21)  # 10, 20, ..., 200
        ]
        self._write_log(log_file, records)

        result = compute_latency_percentiles(log_file)
        assert "step_a" in result
        assert result["step_a"]["count"] == 20
        assert result["step_a"]["p50_ms"] == 105.0  # median of 10..200
        assert result["step_a"]["p95_ms"] == 200.0

    def test_ignores_events_without_duration(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        records = [
            {"event": "no_dur", "payload": {}},
            {"event": "has_dur", "duration_ms": 50.0},
        ]
        self._write_log(log_file, records)

        result = compute_latency_percentiles(log_file)
        assert "no_dur" not in result
        assert "has_dur" in result

    def test_empty_file_returns_empty(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        self._write_log(log_file, [])
        assert compute_latency_percentiles(log_file) == {}

    def test_missing_file_returns_empty(self, tmp_path):
        assert compute_latency_percentiles(str(tmp_path / "nonexistent.jsonl")) == {}

    def test_multiple_event_types(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        records = [
            {"event": "fast", "duration_ms": 10},
            {"event": "fast", "duration_ms": 20},
            {"event": "slow", "duration_ms": 500},
            {"event": "slow", "duration_ms": 1000},
        ]
        self._write_log(log_file, records)

        result = compute_latency_percentiles(log_file)
        assert result["fast"]["count"] == 2
        assert result["slow"]["count"] == 2
        assert result["fast"]["p50_ms"] == 15.0
        assert result["slow"]["p50_ms"] == 750.0
