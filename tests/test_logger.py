"""
Unit tests for the structured trace logger (Steps 13 & 14).
Covers: generate_trace_id, log_event with trace fields, Span, compute_latency_percentiles,
        log_token_usage, compute_cost_summary.
"""
import json
import os
import time
import tempfile

import pytest
from unittest.mock import patch, MagicMock

from agentforge.logger import (
    generate_trace_id, log_event, Span, compute_latency_percentiles,
    log_token_usage, compute_cost_summary, compute_trace_cost, MODEL_COSTS,
)


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


class TestLogTokenUsage:

    def _make_response(self, prompt_tokens=100, completion_tokens=50, model="gpt-4o-mini"):
        resp = MagicMock()
        resp.model = model
        resp.usage.prompt_tokens = prompt_tokens
        resp.usage.completion_tokens = completion_tokens
        return resp

    def test_logs_token_event(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        resp = self._make_response()
        with patch("agentforge.logger.AGENT_LOG_FILE", log_file):
            log_token_usage(resp, "test_op", trace_id="t1")
        with open(log_file, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert record["event"] == "token_usage"
        assert record["trace_id"] == "t1"
        p = record["payload"]
        assert p["operation"] == "test_op"
        assert p["prompt_tokens"] == 100
        assert p["completion_tokens"] == 50
        assert p["total_tokens"] == 150
        assert p["model"] == "gpt-4o-mini"
        assert p["cost_usd"] > 0

    def test_cost_calculation_matches_model_pricing(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        resp = self._make_response(prompt_tokens=1000, completion_tokens=500, model="gpt-4o-mini")
        with patch("agentforge.logger.AGENT_LOG_FILE", log_file):
            log_token_usage(resp, "calc_test")
        with open(log_file, encoding="utf-8") as f:
            record = json.loads(f.readline())
        costs = MODEL_COSTS["gpt-4o-mini"]
        expected = (1000 * costs["prompt"]) + (500 * costs["completion"])
        assert record["payload"]["cost_usd"] == round(expected, 6)

    def test_skips_when_no_usage(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        resp = MagicMock(spec=[])  # no .usage attribute
        with patch("agentforge.logger.AGENT_LOG_FILE", log_file):
            log_token_usage(resp, "noop")
        assert not os.path.exists(log_file)

    def test_model_override(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        resp = self._make_response(model="gpt-4o-mini")
        with patch("agentforge.logger.AGENT_LOG_FILE", log_file):
            log_token_usage(resp, "override_test", model="gpt-4o")
        with open(log_file, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert record["payload"]["model"] == "gpt-4o"

    def test_unknown_model_falls_back_to_gpt4o_mini(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        resp = self._make_response(model="future-model-99")
        with patch("agentforge.logger.AGENT_LOG_FILE", log_file):
            log_token_usage(resp, "fallback_test")
        with open(log_file, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert record["payload"]["cost_usd"] > 0


class TestComputeCostSummary:

    def _write_log(self, path, records):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_aggregates_by_operation(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        records = [
            {"event": "token_usage", "payload": {
                "operation": "intent", "model": "m", "prompt_tokens": 100,
                "completion_tokens": 50, "total_tokens": 150, "cost_usd": 0.001}},
            {"event": "token_usage", "payload": {
                "operation": "intent", "model": "m", "prompt_tokens": 200,
                "completion_tokens": 100, "total_tokens": 300, "cost_usd": 0.002}},
            {"event": "token_usage", "payload": {
                "operation": "generate", "model": "m", "prompt_tokens": 500,
                "completion_tokens": 200, "total_tokens": 700, "cost_usd": 0.005}},
        ]
        self._write_log(log_file, records)
        result = compute_cost_summary(log_file)

        assert result["by_operation"]["intent"]["calls"] == 2
        assert result["by_operation"]["intent"]["prompt_tokens"] == 300
        assert result["by_operation"]["generate"]["calls"] == 1
        assert result["total"]["calls"] == 3
        assert result["total"]["cost_usd"] == 0.008

    def test_ignores_non_token_events(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        records = [
            {"event": "trace_start", "payload": {}},
            {"event": "token_usage", "payload": {
                "operation": "x", "model": "m", "prompt_tokens": 10,
                "completion_tokens": 5, "total_tokens": 15, "cost_usd": 0.0001}},
        ]
        self._write_log(log_file, records)
        result = compute_cost_summary(log_file)
        assert result["total"]["calls"] == 1

    def test_empty_file(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        self._write_log(log_file, [])
        result = compute_cost_summary(log_file)
        assert result["total"]["calls"] == 0

    def test_missing_file(self, tmp_path):
        result = compute_cost_summary(str(tmp_path / "nonexistent.jsonl"))
        assert result["total"]["calls"] == 0
        assert result["by_operation"] == {}


class TestComputeTraceCost:

    def _write_log(self, path, records):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_filters_by_trace_id(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        records = [
            {"event": "token_usage", "trace_id": "aaa", "payload": {
                "operation": "intent", "prompt_tokens": 100, "completion_tokens": 50, "cost_usd": 0.001}},
            {"event": "token_usage", "trace_id": "bbb", "payload": {
                "operation": "generate", "prompt_tokens": 500, "completion_tokens": 200, "cost_usd": 0.005}},
            {"event": "token_usage", "trace_id": "aaa", "payload": {
                "operation": "generate", "prompt_tokens": 400, "completion_tokens": 100, "cost_usd": 0.003}},
        ]
        self._write_log(log_file, records)
        result = compute_trace_cost("aaa", log_file)
        assert result["calls"] == 2
        assert result["prompt_tokens"] == 500
        assert result["completion_tokens"] == 150
        assert result["cost_usd"] == 0.004
        assert result["operations"] == ["intent", "generate"]

    def test_returns_zero_for_unknown_trace(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        self._write_log(log_file, [
            {"event": "token_usage", "trace_id": "xxx", "payload": {
                "operation": "op", "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.0001}},
        ])
        result = compute_trace_cost("nonexistent", log_file)
        assert result["calls"] == 0
        assert result["operations"] == []

    def test_missing_file(self, tmp_path):
        result = compute_trace_cost("any", str(tmp_path / "nope.jsonl"))
        assert result["calls"] == 0
