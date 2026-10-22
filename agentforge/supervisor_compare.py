"""Supervisor setting ON vs OFF, end to end (Step 21.2 part 2b).

The planner eval (``evaluation.py --planner``) scores only the PLAN. This
module runs the WHOLE turn — ``main.run_agent`` with guardrails, workers and
real MCP tools — twice per case: once with ``AGENT_SUPERVISOR_ENABLED`` off
(today's intent classifier) and once with it on. Each turn records:

    correctness  — an LLM judge (``evaluation.score_task_completion``) checks the
                   reply against the case's ``expected_outcome``
    cost         — every LLM call of the turn (``logger.compute_trace_cost``)
    latency      — wall-clock time of the whole turn

The two arms run INTERLEAVED (case 1 off, case 1 on, case 2 on, case 2 off, …)
so live tool results — news, weather — are as close in time as possible for
both arms, and neither arm always runs first.

State is isolated per run: logs, tool pins and the corpus point at temporary
files, and each arm has its OWN memory folder. Cases run in dataset order, so
"what food do I like?" can only pass if the arm really stored "I prefer
vegetarian food" earlier — an end-to-end check, not an accident. The corpus is
a copy of ``tests/fixtures/test_corpus.json`` (the handbook, pre-embedded; the
same file CI's eval gate uses), so a fresh clone can run this.

Gated tools (``fetch``) are auto-approved in BOTH arms: this measures what each
setting can do, not how safe it is — that is the red-team run's job.

Known limits, to state next to any number from here: the judge is the same
small model as the agent; live tool results still differ between arms; one
run is noisy.

    python -m agentforge.supervisor_compare [--runs N] [--cases plan_001,plan_014] [--out run.json]
"""
import argparse
import contextlib
import functools
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from agentforge.evaluation import load_planner_dataset, score_task_completion

logger = logging.getLogger(__name__)

ARMS = ("off", "on")
USER_ID = "supervisor_compare"
# The first full run used the judge's default sampling, and it passed one reply
# while failing an almost identical one ("The provided chunks do not contain
# any information about adoption"). 0 makes the same reply get the same verdict.
JUDGE_TEMPERATURE = 0

_TEST_CORPUS = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "test_corpus.json"


def _auto_approve(_request):
    return True  # approve every gated call, in both arms (see module docstring)


@contextlib.contextmanager
def _isolated_state(tmpdir: str):
    """Point logs, tool pins and the corpus at ``tmpdir``; restore them afterwards.

    Rebinds the module-level names the running code reads (each module bound
    the config value at import time) — the same approach as
    ``redteam_fullstack._isolated_state``, plus the corpus. Memory is set per
    arm by the caller. Yields the log path.
    """
    from agentforge import logger as logger_mod
    from agentforge import main
    from agentforge import mcp_client
    from agentforge.memory import semantic
    from agentforge.rag import document_store

    saved = {
        "log": logger_mod.AGENT_LOG_FILE,
        "pins": mcp_client.AGENT_TOOL_PINS_FILE,
        "corpus": document_store.AGENT_CORPUS_FILE,
        "mem": semantic.MEMORY_DIR,
        "supervisor": main.AGENT_SUPERVISOR_ENABLED,
    }
    log_path = str(Path(tmpdir) / "compare_logs.jsonl")
    corpus_path = str(Path(tmpdir) / "corpus.json")
    shutil.copyfile(_TEST_CORPUS, corpus_path)  # a copy: the run must never write the fixture
    logger_mod.AGENT_LOG_FILE = log_path
    mcp_client.AGENT_TOOL_PINS_FILE = str(Path(tmpdir) / "tool_pins.json")
    document_store.AGENT_CORPUS_FILE = corpus_path
    try:
        yield log_path
    finally:
        logger_mod.AGENT_LOG_FILE = saved["log"]
        mcp_client.AGENT_TOOL_PINS_FILE = saved["pins"]
        document_store.AGENT_CORPUS_FILE = saved["corpus"]
        semantic.MEMORY_DIR = saved["mem"]
        main.AGENT_SUPERVISOR_ENABLED = saved["supervisor"]


def _set_arm(arm: str, memory_dirs: dict) -> None:
    """Switch the running agent to one arm: the supervisor setting and that arm's memory."""
    from agentforge import main
    from agentforge.memory import semantic

    main.AGENT_SUPERVISOR_ENABLED = arm == "on"
    semantic.MEMORY_DIR = memory_dirs[arm]


# Answered on the classifier path by ANSWER, so the turn loads the input AND
# the output guardrail models (an IGNORE turn like "hi" skips the output scan).
WARM_UP_MESSAGE = "explain what a hash map is"


def _warm_up(agent: Callable, tmpdir: str) -> None:
    """One untimed, unrecorded turn before the measured ones.

    The first turn of a process loads the local guardrail models and discovers
    the MCP tools — measured at about five minutes in the first smoke run. That
    cost would land on whichever arm ran first, so it is paid here instead.
    """
    from agentforge import main
    from agentforge.memory import semantic

    main.AGENT_SUPERVISOR_ENABLED = False
    semantic.MEMORY_DIR = str(Path(tmpdir) / "memory_warm_up")
    Path(semantic.MEMORY_DIR).mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        agent(USER_ID, "compare-warm-up", WARM_UP_MESSAGE, history=[], stream=False,
              approval_handler=_auto_approve)
    except Exception as exc:  # a failed warm-up only means the first turn is slower
        logger.warning("supervisor_compare: warm-up turn raised: %s", exc)
    logger.info("supervisor_compare: warm-up took %.0f ms (not measured)",
                (time.perf_counter() - start) * 1000)


def _judge_task(entry: dict) -> str:
    """The task as the judge sees it: the message, plus the conversation before it."""
    history = entry.get("history") or []
    if not history:
        return entry["message"]
    lines = [f"{m['role']}: {m['content']}" for m in history]
    return ("Conversation so far:\n" + "\n".join(lines)
            + f"\n\nThe user's new message (the task): {entry['message']}")


def _run_turn(entry: dict, arm: str, run: int, log_path: str, agent: Callable,
              judge: Callable) -> dict:
    """One case through one arm: run the agent, then read its cost and judge its reply.

    An agent that raises is recorded as an incomplete turn with its error —
    one bad case must not cancel the run.
    """
    from agentforge.logger import compute_trace_cost, generate_trace_id

    tid = generate_trace_id()
    error = None
    start = time.perf_counter()
    try:
        reply = agent(USER_ID, f"compare-{arm}-{run}", entry["message"],
                      history=list(entry.get("history") or []), stream=False,
                      trace_id=tid, approval_handler=_auto_approve)
    except Exception as exc:  # incl. ApprovalRequired, which the auto-approver should prevent
        logger.error("supervisor_compare: %s (%s arm) raised: %s", entry["id"], arm, exc)
        reply, error = "", f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - start) * 1000
    if not isinstance(reply, str):  # stream=False should always give text; be safe
        reply = "".join(reply)

    cost = compute_trace_cost(tid, log_path)
    if error:
        completion = {"completed": False, "reason": f"agent error: {error}", "method": "none"}
    else:
        completion = judge(_judge_task(entry), reply, entry["expected_outcome"])

    return {
        "id": entry["id"],
        "run": run,
        "arm": arm,
        "group": entry.get("group", "unknown"),
        "trace_id": tid,
        "completed": bool(completion["completed"]),
        "judge_reason": completion.get("reason", ""),
        "reply": reply,
        "llm_calls": cost["calls"],
        "tokens": cost["prompt_tokens"] + cost["completion_tokens"],
        "cost_usd": cost["cost_usd"],
        "latency_ms": round(latency_ms, 1),
        "error": error,
    }


def _summarise(turns: list) -> dict:
    """Totals for one arm's turns: completion, tokens, cost, LLM calls, latency."""
    n = len(turns)
    latencies = sorted(t["latency_ms"] for t in turns)
    completed = sum(t["completed"] for t in turns)
    tokens = sum(t["tokens"] for t in turns)
    return {
        "total": n,
        "completed": completed,
        "completion_rate": completed / n if n else 0.0,
        "tokens": tokens,
        "tokens_mean": tokens / n if n else 0.0,
        "cost_usd": round(sum(t["cost_usd"] for t in turns), 6),
        "llm_calls_mean": sum(t["llm_calls"] for t in turns) / n if n else 0.0,
        "latency_ms_mean": sum(latencies) / n if n else 0.0,
        "latency_ms_p50": latencies[n // 2] if n else 0.0,
        "errors": sum(1 for t in turns if t["error"]),
    }


def compare(
    dataset: list,
    runs: int = 1,
    agent: Optional[Callable] = None,
    judge: Optional[Callable] = None,
    tmpdir: Optional[str] = None,
    verbose: bool = True,
    warm_up: bool = True,
) -> dict:
    """Run every case through both arms ``runs`` times and compare them.

    ``agent`` defaults to ``main.run_agent`` and ``judge`` to the LLM judge;
    tests pass fakes. Every case needs ``expected_outcome``. ``warm_up`` runs
    one unrecorded turn first (see ``_warm_up``).

    Returns {arms: {off, on}, by_group: {group: {off, on}}, disagreements,
    runs, per_turn}. A disagreement is a case and run where one arm completed
    the task and the other did not — the part worth reading.
    """
    missing = [e["id"] for e in dataset if not e.get("expected_outcome")]
    if missing:
        raise ValueError(f"cases without expected_outcome: {missing}")
    if agent is None:
        from agentforge.main import run_agent as agent
    if judge is None:
        judge = functools.partial(score_task_completion, temperature=JUDGE_TEMPERATURE)

    with contextlib.ExitStack() as stack:
        if tmpdir is None:
            tmpdir = stack.enter_context(tempfile.TemporaryDirectory(prefix="sup_compare_"))
        log_path = stack.enter_context(_isolated_state(tmpdir))
        if warm_up:
            _warm_up(agent, tmpdir)

        per_turn = []
        for run in range(runs):
            # A fresh memory per arm per run: each run replays the cases in order.
            memory_dirs = {}
            for arm in ARMS:
                memory_dirs[arm] = str(Path(tmpdir) / f"memory_{arm}_{run}")
                Path(memory_dirs[arm]).mkdir(parents=True, exist_ok=True)
            for i, entry in enumerate(dataset):
                order = ARMS if i % 2 == 0 else tuple(reversed(ARMS))
                for arm in order:
                    _set_arm(arm, memory_dirs)
                    turn = _run_turn(entry, arm, run, log_path, agent, judge)
                    per_turn.append(turn)
                    if verbose:
                        status = "DONE" if turn["completed"] else "INCOMPLETE"
                        print(f"[run {run + 1}/{runs}] [{i + 1}/{len(dataset)}] "
                              f"{entry['id']} {arm:3s}: {status}  "
                              f"{turn['tokens']} tok, {turn['latency_ms']:.0f} ms")

    arms = {arm: _summarise([t for t in per_turn if t["arm"] == arm]) for arm in ARMS}
    groups = sorted({t["group"] for t in per_turn})
    by_group = {
        g: {arm: _summarise([t for t in per_turn if t["group"] == g and t["arm"] == arm])
            for arm in ARMS}
        for g in groups
    }
    paired = {}
    for t in per_turn:
        paired.setdefault((t["id"], t["run"]), {})[t["arm"]] = t
    disagreements = [
        {"id": key[0], "run": key[1], "winner": "on" if pair["on"]["completed"] else "off",
         "off": pair["off"], "on": pair["on"]}
        for key, pair in paired.items()
        if pair["on"]["completed"] != pair["off"]["completed"]
    ]
    logger.info(
        "supervisor_compare: completed off %d/%d, on %d/%d; tokens off %d, on %d; "
        "%d disagreement(s)",
        arms["off"]["completed"], arms["off"]["total"],
        arms["on"]["completed"], arms["on"]["total"],
        arms["off"]["tokens"], arms["on"]["tokens"], len(disagreements),
    )
    return {"config": _run_config(), "arms": arms, "by_group": by_group,
            "disagreements": disagreements, "runs": runs, "per_turn": per_turn}


def _run_config() -> dict:
    """The settings a number from this run depends on, saved with it."""
    from agentforge import config

    names = ("OPENAI_MODEL", "AGENT_MODEL_ROUTING_ENABLED", "MODEL_TIER_SMALL",
             "MODEL_TIER_FRONTIER", "AGENT_GUARDRAIL_ENABLED", "AGENT_INPUT_GUARDRAIL_ENABLED",
             "AGENT_OUTPUT_GUARDRAIL_ENABLED", "AGENT_OUTPUT_GUARDRAIL_ENGINE",
             "AGENT_SUPERVISOR_MAX_WORKERS")
    return {n: getattr(config, n) for n in names} | {"JUDGE_TEMPERATURE": JUDGE_TEMPERATURE}


def _short(text: str, limit: int = 300) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + " …"


def _print_report(result: dict) -> None:
    off, on = result["arms"]["off"], result["arms"]["on"]
    print(f"\n{'='*64}")
    print("  Settings: " + ", ".join(f"{k}={v}" for k, v in result["config"].items()))
    print(f"  {'':22s}{'OFF (classifier)':>20s}{'ON (supervisor)':>20s}")
    rows = [
        ("Completed", f"{off['completed']}/{off['total']} ({off['completion_rate']:.0%})",
         f"{on['completed']}/{on['total']} ({on['completion_rate']:.0%})"),
        ("Tokens per turn", f"{off['tokens_mean']:.0f}", f"{on['tokens_mean']:.0f}"),
        ("LLM calls per turn", f"{off['llm_calls_mean']:.1f}", f"{on['llm_calls_mean']:.1f}"),
        ("Cost (total, USD)", f"{off['cost_usd']:.4f}", f"{on['cost_usd']:.4f}"),
        ("Latency mean (ms)", f"{off['latency_ms_mean']:.0f}", f"{on['latency_ms_mean']:.0f}"),
        ("Latency p50 (ms)", f"{off['latency_ms_p50']:.0f}", f"{on['latency_ms_p50']:.0f}"),
        ("Agent errors", f"{off['errors']}", f"{on['errors']}"),
    ]
    for label, a, b in rows:
        print(f"  {label:22s}{a:>20s}{b:>20s}")
    print("\n  Completed by group (off / on):")
    for group, g in result["by_group"].items():
        print(f"    {group:24s}: {g['off']['completed']} / {g['on']['completed']}"
              f"  (n={g['off']['total']})")
    print(f"{'='*64}")

    # Read the disagreements — eval-honesty rule: never report only the headline.
    if result["disagreements"]:
        print(f"\n  {len(result['disagreements'])} case(s) where the arms disagree:")
        for d in result["disagreements"]:
            print(f"\n    {d['id']} (run {d['run'] + 1}) — only {d['winner'].upper()} completed it")
            for arm in ARMS:
                t = d[arm]
                print(f"      {arm:3s} judge: {_short(t['judge_reason'], 200)}")
                print(f"      {arm:3s} reply: {_short(t['reply'])}")
    else:
        print("\n  The arms agree on every case.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Replies are printed and can hold emoji ("Got it 👍"); a cp1252 Windows
    # console would raise UnicodeEncodeError and end the run mid-way.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(
        description="Run every planner case end to end with the supervisor OFF and ON.")
    parser.add_argument("--runs", type=int, default=1, help="repeat the whole set N times")
    parser.add_argument("--cases", type=str, default=None,
                        help="comma-separated case ids (default: all)")
    parser.add_argument("--out", type=str, default=None,
                        help="save the whole run (every turn and reply) as JSON")
    args = parser.parse_args(argv)

    dataset = load_planner_dataset()
    if args.cases:
        wanted = set(args.cases.split(","))
        unknown = wanted - {e["id"] for e in dataset}
        if unknown:
            print(f"Unknown case id(s): {sorted(unknown)}")
            return 2
        dataset = [e for e in dataset if e["id"] in wanted]

    print(f"\nComparing the supervisor OFF vs ON: {len(dataset)} case(s) x {args.runs} run(s) "
          f"= {2 * len(dataset) * args.runs} turns (live API, auto-approved tools)\n")
    result = compare(dataset, runs=args.runs)
    _print_report(result)

    # Every turn of this run, so later numbers and charts come from the same run.
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved every turn of this run to {args.out}")
    return 0  # a measurement, not a gate


if __name__ == "__main__":
    sys.exit(main())
