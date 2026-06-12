# agent/reasoning/react_engine.py

import json
import logging
from openai import OpenAI

from agentforge.approval import (
    ApprovalRequired,
    make_resume_handler,
    run_interruptible,
)
from agentforge.config import OPENAI_MODEL, OPENAI_BASE_URL
from agentforge.prompts import build_prompt, render_tool_catalog, OUTPUT_SCHEMA
from agentforge.mcp_client import mcp_gateway
from agentforge.memory.semantic import get_relevant_memories, store_memory
from agentforge.logger import log_event, log_token_usage

_client = None  # created on first API call, not at import time

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()
    return _client

logger = logging.getLogger(__name__)


def react_loop(user_id: str, user_input: str, max_steps: int = 5,
               approval_handler=None, trace_id: str = None) -> str:
    """Core ReAct reasoning loop (think → act → observe → repeat).

    Public entry point. Existing positional args are unchanged, so callers
    (main.py) are unaffected; internally it runs an async loop because tools are
    reached over MCP. One MCP gateway is opened for the whole turn (see
    ``_react_loop_async``). ``approval_handler`` is the front-end's
    human-in-the-loop callback (Step 17f), passed through to the gateway.
    ``trace_id`` is the turn's trace ID from run_agent (issue #7): without it,
    every log record this module emits — token usage, react_* events, the
    gateway's security audit entries — is orphaned from its turn, so per-trace
    cost aggregation undercounts REACT turns.
    """
    return run_interruptible(_react_loop_async(user_id, user_input, max_steps,
                                               approval_handler=approval_handler,
                                               trace_id=trace_id))


def resume_react_loop(interrupt: ApprovalRequired, decision,
                      approval_handler=None) -> str:
    """Resume a ReAct turn that was interrupted for human approval (Step 17f).

    ``interrupt`` is the ApprovalRequired the front-end caught — its
    ``continuation`` holds the frozen loop state. ``decision`` is the human's
    Deny (False) / Allow once (True) / APPROVE_TURN (allow this tool for the
    rest of the turn — issue #6) for the stored pending call. Resume — not
    replay — so the decision settles the EXACT call the human looked at; the
    LLM is never asked to regenerate it (it wouldn't reproduce the same
    arguments — LLM output is non-deterministic). A Deny resumes too: the model
    receives the "user declined" observation and adapts.
    """
    return run_interruptible(
        _react_resume_async(interrupt, decision, approval_handler))


async def _react_loop_async(user_id: str, user_input: str, max_steps: int = 5,
                            approval_handler=None, trace_id: str = None) -> str:
    """Async ReAct loop with tools served over MCP (Step 17c.2).

    What changed from the pre-MCP loop:
    - Tools are no longer dispatched in-process via a local registry. We open one
      shared MCP gateway for the whole turn and call ``gw.call(...)`` for each
      tool step, so every reasoning step reuses the same warm server sessions.
    - The available tools are discovered at runtime and rendered into the prompt
      (render_tool_catalog), instead of a hardcoded tool name. The agent has zero
      hardcoded tools — it can only call what the servers actually expose.

    Each step uses response_format={"type": "json_object"} (constrained decoding)
    so the API guarantees valid JSON on every step. A single JSONDecodeError
    anywhere in the loop kills the whole reasoning chain — there is no partial
    recovery mid-loop — so the json.JSONDecodeError guard is kept as a last resort.
    Shape validation (action.get("type") etc.) stays too: valid JSON does not mean
    correct keys or values.
    """
    memory_chunks = get_relevant_memories(user_id, user_input)
    log_event("react_start", {"user_input": user_input, "max_steps": max_steps},
              trace_id=trace_id)

    async with mcp_gateway(trace_id, approval_handler=approval_handler) as gw:
        # Build the prompt once, injecting the live tool catalog so the model can
        # only reference tools that are actually served this turn.
        messages = build_prompt(user_input, memory_chunks)
        messages.append({"role": "system", "content": render_tool_catalog(gw.catalog)})
        messages.append({"role": "system", "content": OUTPUT_SCHEMA})
        log_event("react_tools_discovered", {"tools": [t["name"] for t in gw.catalog]},
                  trace_id=trace_id)

        return await _react_steps(gw, messages, user_id, 0, max_steps,
                                  trace_id=trace_id)


async def _react_resume_async(interrupt: ApprovalRequired, decision,
                              fallback=None) -> str:
    """Re-enter an interrupted ReAct turn mid-flight (Step 17f resume).

    Opens a fresh gateway (the original's server subprocesses died when the
    interrupt unwound the turn), settles the STORED pending call with the
    human's decision via a one-shot handler, appends the resulting observation
    to the frozen messages, and continues the loop from the next step. Later
    gated calls go to ``fallback`` (the front-end's normal handler) and can
    interrupt again — each gets its own card.
    """
    cont = interrupt.continuation
    trace_id = cont.get("trace_id")  # frozen with the turn at interrupt time
    handler = make_resume_handler(decision, interrupt.request, fallback)
    log_event("react_resume", {
        "step": cont["step"] + 1,
        "tool": cont["tool_name"],
        "decision": "approved" if decision else "denied",
    }, trace_id=trace_id)

    async with mcp_gateway(trace_id, approval_handler=handler,
                           granted=cont.get("granted")) as gw:
        # Finish the interrupted step: dispatch (or deny) the stored call.
        # The one-shot handler answers it without raising; gw.call's normal
        # contract applies (wrap on success, declined-string on deny).
        try:
            observation = await gw.call(cont["tool_name"], cont["tool_input"])
        except ApprovalRequired as exc:
            exc.continuation = cont  # defensive: keep the turn resumable
            raise

        messages = cont["messages"]
        messages.append({"role": "assistant", "content": cont["raw"]})
        messages.append({
            "role": "user",
            "content": f"Observation from tool '{cont['tool_name']}':\n{observation}"
        })
        return await _react_steps(gw, messages, cont["user_id"],
                                  cont["step"] + 1, cont["max_steps"],
                                  trace_id=trace_id)


async def _react_steps(gw, messages: list, user_id: str,
                       start_step: int, max_steps: int,
                       trace_id: str = None) -> str:
    """The think → act → observe loop body, shared by fresh runs and resumes.

    ``start_step`` is 0 for a fresh turn; a resume passes the interrupted
    step + 1 (the interrupted step itself is finished by the resume path
    before re-entering here).
    """
    for step in range(start_step, max_steps):
            try:
                response = _get_client().chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    # Constrained decoding: same as classify_intent (Step 6).
                    # The ReAct schema is more complex (nested action object,
                    # multiple fields) so parse failures are more likely without
                    # this — and more costly, since they abort a reasoning chain.
                    response_format={"type": "json_object"},
                )
                log_token_usage(response, f"react_step_{step + 1}",
                                trace_id=trace_id)
                raw = response.choices[0].message.content
            except Exception as e:
                logger.error("react_loop: LLM call failed on step %d: %s", step + 1, e, exc_info=True)
                return "I ran into an error during planning. Please try again."

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # Should be extremely rare now that response_format is set.
                # Kept as a safety net for empty or malformed responses.
                logger.warning("react_loop: JSONDecodeError on step %d despite response_format.", step + 1)
                return "Agent error: Invalid JSON response."

            thought = data.get("thought", "")
            action = data.get("action", {})

            print(f"\n[THOUGHT {step+1}] {thought}")
            log_event("react_step", {
                "step": step + 1,
                "thought": thought,
                "action_type": action.get("type"),
                "tool_name": action.get("tool_name"),
            }, trace_id=trace_id)

            # ---------- FINAL ----------
            if action.get("type") == "final":
                if data.get("store_memory") and data.get("memory_text"):
                    store_memory(user_id, data["memory_text"])

                reply = data.get("reply", "")
                log_event("react_end", {"steps_taken": step + 1, "reply_length": len(reply)},
                          trace_id=trace_id)
                return reply

            # ---------- TOOL ----------
            if action.get("type") == "tool":
                tool_name = action.get("tool_name")
                tool_input = action.get("tool_input", {})

                # Dispatch over MCP. gw.call returns a readable error string for
                # unknown/hallucinated tools (it does not raise), so the model can
                # observe the error and recover on the next step. gw.call also
                # wraps the tool output as untrusted data with this turn's nonce
                # (Step 17e) — the spotlight rule in the prompt tells the model to
                # treat anything inside those markers as data, never instructions.
                try:
                    observation = await gw.call(tool_name, tool_input)
                except ApprovalRequired as exc:
                    # CHECKPOINT (Step 17f): the human must decide. Attach
                    # everything resume needs to finish this step and continue
                    # the loop — the loop owns this state, the gateway doesn't.
                    # The exception object travels to the front-end intact.
                    exc.continuation = {
                        "pipeline": "react",
                        "user_id": user_id,
                        "messages": messages,
                        "step": step,
                        "raw": raw,
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "max_steps": max_steps,
                        # Issue #7: freeze the trace ID with the rest of the
                        # turn so the resumed half logs under the same trace
                        # (main.resume_agent reads it back with cont.get).
                        "trace_id": trace_id,
                        # Issue #6: the turn's approval-grant set rides in the
                        # continuation — resume hands it to the next gateway,
                        # so "allow for the rest of this turn" survives every
                        # interrupt and dies with the turn (no cleanup code).
                        "granted": gw.granted,
                    }
                    raise

                # The ReAct loop speaks the prompt-based JSON protocol: the model
                # returns {thought, action} as assistant *content*, not native
                # tool_calls. So the observation must go back as an ordinary
                # conversation turn — a user message — NOT a role:"tool" message.
                # The OpenAI API only accepts a "tool" message immediately after an
                # assistant message that carried tool_calls (the native
                # function-calling path); feeding an orphan "tool" message here
                # makes the next request fail with a 400. (Issue #1.)
                #
                # `observation` is already wrapped as <untrusted_data_<nonce>> by
                # gw.call, so spotlighting (Step 17e) still holds inside the user
                # turn — the spotlight rules tell the model to treat anything
                # between those markers as data, never instructions.
                messages.append({
                    "role": "assistant",
                    "content": raw
                })

                messages.append({
                    "role": "user",
                    "content": f"Observation from tool '{tool_name}':\n{observation}"
                })

    log_event("react_end", {"steps_taken": max_steps, "reply_length": 0, "stopped": "max_steps"},
              trace_id=trace_id)
    return "Agent stopped: too many reasoning steps."
