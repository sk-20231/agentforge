"""Tests for the supervisor / planner-worker orchestration (Step 21.2).

All LLM calls and MCP servers are faked — no API key, no subprocesses. The
pause/resume tests run the REAL ReAct graph nested inside the REAL supervisor
graph, with a fake LLM and a fake gateway whose context manager wraps
exceptions in an ExceptionGroup, as the real gateway's anyio task group does.
"""
import contextlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentforge import supervisor
from agentforge.approval import ApprovalRequest, ApprovalRequired
from agentforge.logger import log_token_usage, token_meter


# ------------------------------------ helpers ------------------------------------

def _tool_call(name, brief=None, raw_args=None):
    args = raw_args if raw_args is not None else json.dumps({"brief": brief})
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=args))


def _planner_response(calls=None, content=None):
    msg = SimpleNamespace(tool_calls=calls, content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


def _chat_response(content, prompt_tokens=10, completion_tokens=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        model="gpt-4o-mini")


def _events(name):
    import agentforge.logger as lg
    try:
        with open(lg.AGENT_LOG_FILE, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
    except FileNotFoundError:
        return []
    return [r for r in records if r.get("event") == name]


@pytest.fixture
def planner_llm(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(supervisor, "_client", client)
    return client.chat.completions.create


@pytest.fixture
def combine_llm(monkeypatch):
    """The supervisor's own client — used by plan_turn and combine."""
    client = MagicMock()
    client.chat.completions.create.return_value = _chat_response("COMBINED")
    monkeypatch.setattr(supervisor, "_client", client)
    return client.chat.completions.create


# ------------------------------------ plan_turn ------------------------------------

class TestPlanTurn:
    def test_parallel_tool_calls_become_the_plan(self, planner_llm):
        planner_llm.return_value = _planner_response([
            _tool_call("read_user_documents", "Parental leave in the handbook"),
            _tool_call("research", "Latest news on parental leave laws"),
        ])
        plan = supervisor.plan_turn("q", history=[{"role": "user", "content": "earlier"}])
        assert plan.tasks == [
            {"worker": "read_user_documents", "brief": "Parental leave in the handbook"},
            {"worker": "research", "brief": "Latest news on parental leave laws"},
        ]
        assert plan.direct_reply is None
        kwargs = planner_llm.call_args.kwargs
        assert kwargs["temperature"] == 0
        assert [t["function"]["name"] for t in kwargs["tools"]] == list(
            supervisor.SUPERVISOR_WORKER_NAMES)
        # History reaches the planner (to resolve "there"/"it"), before the request.
        assert kwargs["messages"][1] == {"role": "user", "content": "earlier"}
        assert kwargs["messages"][-1] == {"role": "user", "content": "q"}

    def test_no_tool_call_is_a_direct_reply(self, planner_llm):
        planner_llm.return_value = _planner_response(None, content=" Hello! ")
        plan = supervisor.plan_turn("hi")
        assert plan.tasks == [] and plan.direct_reply == "Hello!"

    def test_no_tool_call_and_no_text_uses_the_fallback_reply(self, planner_llm):
        planner_llm.return_value = _planner_response(None, content=None)
        assert supervisor.plan_turn("hi").direct_reply == supervisor.DIRECT_REPLY_FALLBACK

    @pytest.mark.parametrize("call, reason", [
        (_tool_call("delete_everything", "x"), "unknown_worker"),
        (_tool_call("research", "   "), "empty_brief"),
        (_tool_call("research", raw_args="{not json"), "bad_arguments"),
        (_tool_call("research", raw_args="[1, 2]"), "empty_brief"),
    ])
    def test_an_unusable_call_invalidates_the_whole_plan(self, planner_llm, call, reason):
        planner_llm.return_value = _planner_response([_tool_call("answer", "ok"), call])
        assert supervisor.plan_turn("q") is None
        assert _events("supervisor_plan_invalid")[-1]["payload"]["reason"] == reason

    def test_a_non_function_tool_call_invalidates_the_plan(self, planner_llm):
        planner_llm.return_value = _planner_response([SimpleNamespace(custom={"x": 1})])
        assert supervisor.plan_turn("q") is None
        assert _events("supervisor_plan_invalid")[-1]["payload"]["reason"] == "unknown_worker"

    def test_more_tasks_than_the_cap_is_invalid_not_truncated(self, planner_llm, monkeypatch):
        monkeypatch.setattr(supervisor, "AGENT_SUPERVISOR_MAX_WORKERS", 2)
        planner_llm.return_value = _planner_response(
            [_tool_call("answer", f"part {i}") for i in range(3)])
        assert supervisor.plan_turn("q") is None
        assert _events("supervisor_plan_invalid")[-1]["payload"]["reason"] == "too_many_tasks"

    def test_llm_error_returns_none(self, planner_llm):
        planner_llm.side_effect = RuntimeError("boom")
        assert supervisor.plan_turn("q") is None

    def test_plan_log_holds_brief_lengths_not_brief_text(self, planner_llm):
        planner_llm.return_value = _planner_response(
            [_tool_call("remember", "My card number is 4111")])
        supervisor.plan_turn("q")
        payload = _events("supervisor_plan")[-1]["payload"]
        assert payload["workers"] == ["remember"]
        assert payload["brief_chars"] == [len("My card number is 4111")]
        assert "4111" not in json.dumps(payload)


# --------------------------------- run_agent wiring ---------------------------------

@pytest.fixture
def agent(monkeypatch):
    """run_agent with the supervisor ON and every pipeline mocked."""
    import agentforge.main as main
    monkeypatch.setattr(main, "AGENT_SUPERVISOR_ENABLED", True)
    monkeypatch.setattr(main, "_input_guardrail_block", lambda *a, **k: None)
    monkeypatch.setattr(main, "_scan_output", lambda text, tid=None: f"SCANNED:{text}")
    mocks = SimpleNamespace(
        plan=MagicMock(), classify=MagicMock(return_value={
            "intent": "ANSWER", "memory_candidate": "", "reason": "t"}),
        react=MagicMock(return_value="react reply"),
        docs=MagicMock(return_value="docs reply"),
        answer=MagicMock(return_value="answer reply"),
        store=MagicMock(), multi=MagicMock(return_value="multi reply"))
    monkeypatch.setattr(supervisor, "plan_turn", mocks.plan)
    monkeypatch.setattr(supervisor, "run_multi", mocks.multi)
    monkeypatch.setattr(main, "classify_intent", mocks.classify)
    monkeypatch.setattr(main, "run_react_agent", mocks.react)
    monkeypatch.setattr(main, "answer_from_docs", mocks.docs)
    monkeypatch.setattr(main, "answer_with_memory", mocks.answer)
    monkeypatch.setattr(main, "store_memory", mocks.store)
    mocks.run = lambda text, history=None, stream=False: main.run_agent(
        "u1", "s1", text, history=history, stream=stream, trace_id="t-sup")
    return mocks


class TestRunAgentWiring:
    def test_flag_off_never_plans(self, agent, monkeypatch):
        import agentforge.main as main
        monkeypatch.setattr(main, "AGENT_SUPERVISOR_ENABLED", False)
        assert agent.run("explain RAG") == "SCANNED:answer reply"
        agent.plan.assert_not_called()
        agent.classify.assert_called_once()

    def test_unusable_plan_falls_back_to_the_classifier(self, agent):
        agent.plan.return_value = None
        assert agent.run("explain RAG") == "SCANNED:answer reply"
        agent.classify.assert_called_once()

    def test_unexpected_planner_error_falls_back_to_the_classifier(self, agent):
        agent.plan.side_effect = RuntimeError("tool discovery failed")
        assert agent.run("explain RAG") == "SCANNED:answer reply"
        agent.classify.assert_called_once()

    def test_direct_reply_skips_the_classifier_and_is_scanned(self, agent):
        agent.plan.return_value = supervisor.Plan(tasks=[], direct_reply="Hi there!")
        assert agent.run("hi") == "SCANNED:Hi there!"
        agent.classify.assert_not_called()

    def test_one_research_task_runs_react_with_the_original_message(self, agent):
        agent.plan.return_value = supervisor.Plan(
            tasks=[{"worker": "research", "brief": "Plan a Berlin weekend"}])
        agent.run("Plan my weekend in Berlin around the weather.")
        assert agent.react.call_args.args[1] == "Plan my weekend in Berlin around the weather."
        agent.classify.assert_not_called()

    def test_one_remember_task_stores_the_brief(self, agent):
        agent.plan.return_value = supervisor.Plan(
            tasks=[{"worker": "remember", "brief": "The user is vegetarian."}])
        agent.run("I'm vegetarian")
        agent.store.assert_called_once_with("u1", "The user is vegetarian.")

    def test_one_docs_task_keeps_history_and_streaming(self, agent):
        history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        agent.plan.return_value = supervisor.Plan(
            tasks=[{"worker": "read_user_documents", "brief": "sick days in the handbook"}])
        agent.run("And what about sick days?", history=history, stream=True)
        args, kwargs = agent.docs.call_args
        assert args[0] == "And what about sick days?"
        assert kwargs["history"] == history and kwargs["stream"] is True

    def test_multi_task_plan_runs_the_graph_and_scans_its_reply(self, agent):
        tasks = [{"worker": "answer", "brief": "a"}, {"worker": "research", "brief": "b"}]
        agent.plan.return_value = supervisor.Plan(tasks=tasks)
        assert agent.run("two things") == "SCANNED:multi reply"
        assert agent.multi.call_args.args[:3] == ("u1", "two things", tasks)


# ------------------------------- the graph (no pause) -------------------------------

@pytest.fixture
def workers(monkeypatch):
    m = SimpleNamespace(docs=MagicMock(return_value="Handbook: use approved AI tools."),
                        store=MagicMock())
    monkeypatch.setattr(supervisor, "answer_from_docs", m.docs)
    monkeypatch.setattr(supervisor, "store_memory", m.store)
    return m


def _combine_input(combine_llm):
    return combine_llm.call_args.kwargs["messages"][-1]["content"]


SAVED = "Saved this fact about the user: The user is vegetarian."


class TestSupervisorGraph:
    """Graphs with no research task: no gateway, no REACT."""

    def test_workers_run_in_order_and_combine_writes_the_reply(self, workers, combine_llm):
        tasks = [{"worker": "remember", "brief": "The user is vegetarian."},
                 {"worker": "read_user_documents", "brief": "AI tools in the handbook"}]
        reply = supervisor.run_multi("u1", "I'm vegetarian; what does my handbook say?", tasks)
        assert reply == "COMBINED"
        workers.store.assert_called_once_with("u1", "The user is vegetarian.")
        assert workers.docs.call_args.args[0] == "AI tools in the handbook"
        assert workers.docs.call_args.kwargs["end_trace"] is False  # no mid-turn trace_end
        text = _combine_input(combine_llm)
        assert text.index(SAVED) < text.index("Handbook: use")
        assert "I'm vegetarian; what does my handbook say?" in text

    def test_untrusted_results_are_nonce_wrapped_trusted_are_not(self, workers, combine_llm):
        tasks = [{"worker": "remember", "brief": "The user is vegetarian."},
                 {"worker": "read_user_documents", "brief": "b"}]
        supervisor.run_multi("u1", "q", tasks)
        text = _combine_input(combine_llm)
        assert '<untrusted_data_' in text and 'source="worker:read_user_documents"' in text
        assert 'source="worker:remember"' not in text
        system = [m["content"] for m in combine_llm.call_args.kwargs["messages"]
                  if m["role"] == "system"]
        assert supervisor.SPOTLIGHT_INSTRUCTIONS in system

    def test_a_failed_worker_does_not_stop_the_others(self, workers, combine_llm):
        workers.docs.side_effect = RuntimeError("corpus missing")
        tasks = [{"worker": "read_user_documents", "brief": "b"},
                 {"worker": "remember", "brief": "The user is vegetarian."}]
        assert supervisor.run_multi("u1", "q", tasks) == "COMBINED"
        workers.store.assert_called_once()
        assert "FAILED (RuntimeError)" in _combine_input(combine_llm)
        statuses = [e["payload"]["status"] for e in _events("invoke_agent")]
        assert statuses == ["error", "ok"]

    def test_all_workers_failed_skips_combine(self, workers, combine_llm):
        workers.docs.side_effect = RuntimeError("x")
        workers.store.side_effect = RuntimeError("y")
        tasks = [{"worker": "read_user_documents", "brief": "b"},
                 {"worker": "remember", "brief": "a"}]
        assert supervisor.run_multi("u1", "q", tasks) == supervisor.ALL_FAILED_REPLY
        combine_llm.assert_not_called()

    def test_combine_failure_degrades_to_the_joined_results(self, workers, combine_llm):
        combine_llm.side_effect = RuntimeError("api down")
        tasks = [{"worker": "remember", "brief": "The user is vegetarian."},
                 {"worker": "read_user_documents", "brief": "b"}]
        assert supervisor.run_multi("u1", "q", tasks) == (
            f"{SAVED}\n\nHandbook: use approved AI tools.")

    def test_invoke_agent_events_carry_genai_names_and_tokens(self, workers, combine_llm):
        def docs_spending_tokens(brief, **kwargs):
            log_token_usage(_chat_response("x", 100, 20), "docs_qa_generate")
            return "from the docs"
        workers.docs.side_effect = docs_spending_tokens
        tasks = [{"worker": "read_user_documents", "brief": "b"},
                 {"worker": "remember", "brief": "a"}]
        supervisor.run_multi("u1", "q", tasks, trace_id="t-genai")
        first, second = _events("invoke_agent")
        assert first["payload"]["gen_ai.operation.name"] == "invoke_agent"
        assert first["payload"]["gen_ai.agent.name"] == "read_user_documents"
        assert first["payload"]["tokens_used"] == 120
        assert first["payload"]["trust"] == "untrusted"
        assert second["payload"]["tokens_used"] == 0
        assert second["payload"]["trust"] == "trusted"
        assert first["trace_id"] == "t-genai"

    def test_finished_turn_is_deleted_from_the_checkpointer(self, workers, combine_llm):
        captured = {}
        real = supervisor.new_thread_id

        def capture(pipeline):
            captured["id"] = real(pipeline)
            return captured["id"]
        with patch.object(supervisor, "new_thread_id", capture):
            supervisor.run_multi("u1", "q", [{"worker": "remember", "brief": "a"},
                                             {"worker": "remember", "brief": "b"}])
        assert captured["id"].startswith("sup-")
        with pytest.raises(ValueError):
            supervisor.pending_supervisor_approval(captured["id"])


class TestTrustAndHelpers:
    @pytest.mark.parametrize("worker, tools, expected", [
        ("research", ["get_weather"], "untrusted"),
        ("research", [], "trusted"),
        ("read_user_documents", [], "untrusted"),
        ("remember", [], "trusted"),
    ])
    def test_trust_is_set_by_code(self, worker, tools, expected):
        assert supervisor._trust(worker, tools) == expected

    def test_answer_runs_as_research_in_a_multi_task_plan(self):
        assert supervisor._multi_task({"worker": "answer", "brief": "b"}) == {
            "worker": "research", "brief": "b", "planned_worker": "answer"}
        assert supervisor._multi_task({"worker": "remember", "brief": "b"}) == {
            "worker": "remember", "brief": "b"}

    def test_fewer_than_two_research_tasks_are_not_merged(self):
        tasks = [{"worker": "research", "brief": "a"}, {"worker": "remember", "brief": "b"}]
        assert supervisor._merge_research(tasks, "q") is tasks

    def test_research_tasks_merge_at_the_first_ones_position(self):
        tasks = [{"worker": "remember", "brief": "fact"},
                 {"worker": "research", "brief": "Weather in Paris"},
                 {"worker": "read_user_documents", "brief": "docs"},
                 {"worker": "research", "brief": "Explain RAG", "planned_worker": "answer"}]
        out = supervisor._merge_research(tasks, "the request")
        assert [t["worker"] for t in out] == ["remember", "research", "read_user_documents"]
        merged = out[1]
        assert "1. Weather in Paris\n2. Explain RAG" in merged["brief"]
        assert "The user's request: the request" in merged["brief"]
        # Measured: without this, REACT returned the reply as a JSON object.
        assert "Never write the reply as JSON" in merged["brief"]
        assert merged["planned_worker"] == "research,answer"
        assert merged["merged_parts"] == 2
        assert merged["max_steps"] == supervisor.RESEARCH_STEPS + supervisor.RESEARCH_STEPS_PER_EXTRA_PART

    def test_step_budget_and_recursion_limit_grow_with_merged_parts(self):
        tasks = [{"worker": "research", "brief": str(n)} for n in range(4)]
        (merged,) = supervisor._merge_research(tasks, "q")
        assert merged["max_steps"] == 5 + 2 * 3
        # The nested REACT graph needs 2 * max_steps + 5 super-steps.
        assert supervisor._recursion_limit([merged]) >= 2 * merged["max_steps"] + 5
        assert supervisor._recursion_limit([{"worker": "remember", "brief": "a"}]) == 25

    def test_tools_called_reads_observation_messages(self):
        from agentforge.reasoning.react_engine import tools_called
        messages = [{"role": "user", "content": "question"},
                    {"role": "assistant", "content": "{}"},
                    {"role": "user", "content": "Observation from tool 'get_weather':\nsunny"},
                    {"role": "user", "content": "Observation from tool 'fetch':\npage"}]
        assert tools_called(messages) == ["get_weather", "fetch"]

    def test_token_meter_counts_only_inside_its_block(self):
        log_token_usage(_chat_response("x", 1, 1), "outside")
        with token_meter() as outer:
            log_token_usage(_chat_response("x", 10, 5), "a")
            with token_meter() as inner:
                log_token_usage(_chat_response("x", 100, 50), "b")
        assert inner.total == 150
        assert outer.total == 15


# --------------------------- research workers (real nested REACT) ---------------------------

class _GatedGateway:
    """Fake MCPGateway: every call asks the approval handler first."""

    def __init__(self, executed):
        self.catalog = [{"name": "fetch", "description": "Fetch a URL",
                         "input_schema": {"properties": {"url": {"type": "string"}}}}]
        self.granted = set()
        self.nonce = "n0nce"
        self.executed = executed
        self.approval_handler = None

    @property
    def has_tools(self):
        return True

    async def call(self, tool_name, arguments):
        request = ApprovalRequest(tool=tool_name, server="fetch", arguments=arguments)
        handler = self.approval_handler
        decision = handler(request) if handler is not None else False
        if not decision:
            return "The user declined this tool call."
        self.executed.append(arguments["url"])
        return f"<untrusted_data_n0nce>page {arguments['url']}</untrusted_data_n0nce>"


@pytest.fixture
def gated(monkeypatch):
    """Patch the supervisor's gateway with one whose exit wraps any exception in
    an ExceptionGroup — the real gateway's anyio task group does exactly that."""
    executed, opened = [], []

    @contextlib.asynccontextmanager
    async def fake_gateway(trace_id=None, approval_handler=None):
        gw = _GatedGateway(executed)
        gw.approval_handler = approval_handler
        opened.append(gw)
        try:
            yield gw
        except BaseException as exc:
            raise BaseExceptionGroup("unhandled errors in a TaskGroup", [exc])
    monkeypatch.setattr(supervisor, "mcp_gateway", fake_gateway)
    monkeypatch.setattr("agentforge.reasoning.react_engine.get_relevant_memories",
                        lambda *a, **k: "no memories")
    return SimpleNamespace(executed=executed, opened=opened)


def _fetch_steps(url):
    """One ReAct worker that fetches ``url``: a tool step, then a final step."""
    return [json.dumps({"thought": "fetch it",
                        "action": {"type": "tool", "tool_name": "fetch",
                                   "tool_input": {"url": url}}}),
            json.dumps({"thought": "done", "action": {"type": "final"},
                        "reply": f"summary of {url}"})]


def _answer_steps(text):
    """One ReAct worker that answers from knowledge: a single final step."""
    return [json.dumps({"thought": "I know this", "action": {"type": "final"},
                        "reply": text})]


def _react_llm(monkeypatch, *workers_steps):
    steps = [s for worker in workers_steps for s in worker]
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=s))],
                        usage=None) for s in steps]
    monkeypatch.setattr("agentforge.reasoning.react_engine._client", client)
    return client.chat.completions.create


def _streamlit_handler(request):
    raise ApprovalRequired(request)


class TestResearchWorkers:
    def test_research_and_answer_parts_run_as_one_react_loop(self, gated, workers, combine_llm,
                                                             monkeypatch):
        """A planner swap (lookup given to answer) stays harmless: every part
        runs on REACT — now in ONE loop, with no combine call."""
        think = _react_llm(monkeypatch, _fetch_steps("https://a.example"))
        tasks = [{"worker": "answer", "brief": "Explain RAG"},
                 {"worker": "answer", "brief": "Fetch https://a.example"}]
        reply = supervisor.run_multi("u1", "Explain RAG and fetch a.example", tasks,
                                     approval_handler=lambda r: True)
        assert reply == "summary of https://a.example"   # REACT's reply, not combine's
        combine_llm.assert_not_called()
        assert gated.executed == ["https://a.example"]
        assert think.call_count == 2                       # one loop: tool step + final step
        (event,) = _events("invoke_agent")
        assert event["payload"]["gen_ai.agent.name"] == "research"
        assert event["payload"]["planned_worker"] == "answer,answer"
        assert event["payload"]["trust"] == "untrusted"    # fetch ran
        # The loop got both parts, the user's request, and the extra steps.
        (start,) = _events("react_start")
        assert start["payload"]["max_steps"] == 7
        prompt = json.dumps(think.call_args_list[0].kwargs["messages"])
        assert "1. Explain RAG" in prompt and "2. Fetch https://a.example" in prompt
        assert "Explain RAG and fetch a.example" in prompt
        assert _events("supervisor_combine")[-1]["payload"]["skipped"] == "single_result"

    def test_merged_research_still_combines_with_other_workers(self, gated, workers,
                                                               combine_llm, monkeypatch):
        think = _react_llm(monkeypatch, _answer_steps("Paris 18 C, Tokyo 21 C."))
        tasks = [{"worker": "research", "brief": "Weather in Paris"},
                 {"worker": "remember", "brief": "The user is vegetarian."},
                 {"worker": "research", "brief": "Weather in Tokyo"}]
        assert supervisor.run_multi("u1", "q", tasks) == "COMBINED"
        assert think.call_count == 1                       # one REACT loop for both lookups
        assert [e["payload"]["gen_ai.agent.name"] for e in _events("invoke_agent")] == [
            "research", "remember"]
        text = _combine_input(combine_llm)
        assert text.index("Paris 18 C, Tokyo 21 C.") < text.index(SAVED)
        start = _events("supervisor_start")[-1]["payload"]
        assert start["planned_tasks"] == 3 and start["merged_research_parts"] == 2

    def test_one_gateway_per_turn_and_none_without_research(self, gated, workers, combine_llm,
                                                            monkeypatch):
        _react_llm(monkeypatch, _answer_steps("a"), _answer_steps("b"))
        supervisor.run_multi("u1", "q", [{"worker": "research", "brief": "a"},
                                         {"worker": "answer", "brief": "b"}])
        assert len(gated.opened) == 1
        supervisor.run_multi("u1", "q", [{"worker": "remember", "brief": "a"},
                                         {"worker": "read_user_documents", "brief": "b"}])
        assert len(gated.opened) == 1

    def test_research_pause_resumes_without_rerunning_finished_work(
            self, gated, workers, combine_llm, monkeypatch):
        think = _react_llm(monkeypatch, _fetch_steps("https://a.example"))
        tasks = [{"worker": "read_user_documents", "brief": "AI tools in the handbook"},
                 {"worker": "research", "brief": "Summarise https://a.example"}]

        with pytest.raises(ApprovalRequired) as paused:
            supervisor.run_multi("u1", "q", tasks, trace_id="t-pause",
                                 approval_handler=_streamlit_handler)
        thread_id = paused.value.thread_id
        assert thread_id.startswith("sup-")
        assert gated.executed == []                     # nothing ran before approval

        pending = supervisor.pending_supervisor_approval(thread_id)
        assert pending.request.tool == "fetch"
        assert pending.request.arguments == {"url": "https://a.example"}
        assert pending.user_id == "u1" and pending.trace_id == "t-pause"

        reply = supervisor.resume(thread_id, True, approval_handler=_streamlit_handler)
        assert reply == "COMBINED"
        assert gated.executed == ["https://a.example"]  # the approved call, exactly once
        workers.docs.assert_called_once()               # the finished worker did not re-run
        assert think.call_count == 2                    # ReAct's LLM step did not re-run
        assert len(gated.opened) == 2                   # one gateway per invocation
        # combine gets the worker's reply (not the raw page), wrapped as untrusted.
        text = _combine_input(combine_llm)
        assert 'source="worker:research"' in text
        assert text.index('source="worker:research"') < text.index("summary of https://a.example")
        research = [e for e in _events("invoke_agent")
                    if e["payload"]["gen_ai.agent.name"] == "research"][-1]
        assert research["payload"]["tools_called"] == ["fetch"]
        assert research["payload"]["trust"] == "untrusted"
        with pytest.raises(ValueError):                 # a finished turn is not pending
            supervisor.pending_supervisor_approval(thread_id)

    def test_deny_resumes_without_running_the_tool(self, gated, workers, combine_llm,
                                                   monkeypatch):
        _react_llm(monkeypatch, _fetch_steps("https://a.example"))
        tasks = [{"worker": "research", "brief": "Summarise https://a.example"},
                 {"worker": "remember", "brief": "a"}]
        with pytest.raises(ApprovalRequired) as paused:
            supervisor.run_multi("u1", "q", tasks, approval_handler=_streamlit_handler)
        assert supervisor.resume(paused.value.thread_id, False,
                                 approval_handler=_streamlit_handler) == "COMBINED"
        assert gated.executed == []
        workers.store.assert_called_once()

    def test_a_later_gated_call_in_the_merged_loop_asks_again(self, gated, workers,
                                                              combine_llm, monkeypatch):
        """The resume decision settles ONLY the stored call: the merged loop's
        next gated call goes back to the front-end and gets its own card."""
        fetch_a, fetch_b = _fetch_steps("https://a.example")[0], _fetch_steps("https://b.example")[0]
        final = json.dumps({"thought": "done", "action": {"type": "final"}, "reply": "both pages"})
        _react_llm(monkeypatch, [fetch_a, fetch_b, final])
        tasks = [{"worker": "research", "brief": "a"}, {"worker": "research", "brief": "b"}]
        with pytest.raises(ApprovalRequired) as first:
            supervisor.run_multi("u1", "q", tasks, approval_handler=_streamlit_handler)
        with pytest.raises(ApprovalRequired) as second:
            supervisor.resume(first.value.thread_id, True,
                              approval_handler=_streamlit_handler)
        assert second.value.thread_id == first.value.thread_id
        assert second.value.request.arguments == {"url": "https://b.example"}
        assert gated.executed == ["https://a.example"]
        assert supervisor.resume(second.value.thread_id, True,
                                 approval_handler=_streamlit_handler) == "both pages"
        assert gated.executed == ["https://a.example", "https://b.example"]

    def test_no_handler_denies_by_default(self, gated, workers, combine_llm, monkeypatch):
        _react_llm(monkeypatch, _fetch_steps("https://a.example"))
        tasks = [{"worker": "research", "brief": "a"}, {"worker": "remember", "brief": "b"}]
        assert supervisor.run_multi("u1", "q", tasks) == "COMBINED"
        assert gated.executed == []
        assert gated.opened[0].approval_handler is None

    def test_main_routes_a_sup_thread_to_the_supervisor(self, gated, workers, combine_llm,
                                                        monkeypatch):
        import agentforge.main as main
        monkeypatch.setattr(main, "_scan_output", lambda text, tid=None: f"SCANNED:{text}")
        _react_llm(monkeypatch, _fetch_steps("https://a.example"))
        tasks = [{"worker": "research", "brief": "a"}, {"worker": "remember", "brief": "b"}]
        with pytest.raises(ApprovalRequired) as paused:
            supervisor.run_multi("u1", "q", tasks, approval_handler=_streamlit_handler)
        thread_id = paused.value.thread_id
        assert main.pending_approval(thread_id).pipeline == "sup"
        assert main.resume_agent(thread_id, True,
                                 approval_handler=_streamlit_handler) == "SCANNED:COMBINED"

    def test_discard_drops_a_paused_supervisor_turn(self, gated, workers, combine_llm,
                                                    monkeypatch):
        import agentforge.main as main
        _react_llm(monkeypatch, _fetch_steps("https://a.example"))
        tasks = [{"worker": "research", "brief": "a"}, {"worker": "remember", "brief": "b"}]
        with pytest.raises(ApprovalRequired) as paused:
            supervisor.run_multi("u1", "q", tasks, approval_handler=_streamlit_handler)
        main.discard_approval(paused.value.thread_id, reason="clear_chat")
        with pytest.raises(ValueError):
            main.pending_approval(paused.value.thread_id)
