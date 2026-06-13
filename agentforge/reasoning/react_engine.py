# agent/reasoning/react_engine.py

import json
import logging
from openai import OpenAI

from agentforge.approval import (
    ApprovalRequired,
    make_resume_handler,
    run_interruptible,
)
from agentforge.config import (
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    REACT_OBS_COMPRESS_THRESHOLD,
)
from agentforge.prompts import (
    build_prompt,
    render_tool_catalog,
    OBSERVATION_COMPRESSION_PROMPT,
    OUTPUT_SCHEMA,
    REACT_TOOL_EFFICIENCY,
    SPOTLIGHT_INSTRUCTIONS,
)
from agentforge.mcp_client import mcp_gateway
from agentforge.memory.semantic import get_relevant_memories, store_memory
from agentforge.safety import wrap_untrusted
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
        messages.append({"role": "system", "content": REACT_TOOL_EFFICIENCY})
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
        # Same compression as the in-loop path (issue #8): the approved call's
        # observation is as oversized as any other fetch chunk.
        observation = _maybe_compress_observation(
            observation, messages, gw, trace_id=trace_id)
        messages.append({
            "role": "user",
            "content": f"Observation from tool '{cont['tool_name']}':\n{observation}"
        })
        return await _react_steps(gw, messages, cont["user_id"],
                                  cont["step"] + 1, cont["max_steps"],
                                  trace_id=trace_id)


def _user_question(messages: list) -> str:
    """The original user question in a ReAct message list.

    Observations are also appended as ``user`` messages (issue #1), so the
    question is the first user message that is NOT a tool observation. Works
    on fresh and resumed (frozen) message lists alike — no extra state to
    thread through the continuation.
    """
    for m in messages:
        if m.get("role") == "user" and not str(m.get("content", "")).startswith(
                "Observation from tool"):
            return m["content"]
    return ""


def _maybe_compress_observation(observation: str, messages: list, gw,
                                trace_id: str = None) -> str:
    """Query-focused compression of an oversized observation (issue #8).

    Every observation appended to ``messages`` is re-sent on EVERY later step,
    so a raw 3-8k-char fetch chunk inflates the cost of each remaining step and
    buries the relevant facts mid-context. One extra LLM call here reads the
    chunk ONCE, extracts only what answers the user's question, and the loop
    carries that instead. Below the threshold the observation passes through
    untouched (zero added cost on the common path).

    Security: the chunk is untrusted data, and a hostile page that cannot
    inject the agent must not get to inject the summarizer instead — so the
    compressor receives the same SPOTLIGHT_INSTRUCTIONS, and its output (a
    derivative of untrusted text) is RE-WRAPPED with the turn's nonce before
    it enters the history. Compression failure falls back to the raw
    observation: a logging-adjacent optimization must never kill the turn.
    """
    if REACT_OBS_COMPRESS_THRESHOLD <= 0:
        return observation
    if len(observation) <= REACT_OBS_COMPRESS_THRESHOLD:
        return observation

    question = _user_question(messages)
    try:
        response = _get_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SPOTLIGHT_INSTRUCTIONS},
                {"role": "system",
                 "content": OBSERVATION_COMPRESSION_PROMPT.format(question=question)},
                {"role": "user", "content": observation},
            ],
        )
        log_token_usage(response, "react_obs_compress", trace_id=trace_id)
        summary = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("react: observation compression failed, keeping raw "
                       "observation: %s", e)
        log_event("react_obs_compress_failed", {"error": str(e)},
                  trace_id=trace_id)
        return observation

    if not summary:
        return observation
    log_event("react_obs_compressed", {
        "original_len": len(observation),
        "summary_len": len(summary),
    }, trace_id=trace_id)
    return wrap_untrusted(summary, source="compressed tool output",
                          nonce=gw.nonce)


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

                # Oversized observations are compressed relative to the user's
                # question before entering the history (issue #8) — otherwise
                # every later step re-sends the raw chunk.
                observation = _maybe_compress_observation(
                    observation, messages, gw, trace_id=trace_id)
                messages.append({
                    "role": "user",
                    "content": f"Observation from tool '{tool_name}':\n{observation}"
                })

    log_event("react_end", {"steps_taken": max_steps, "reply_length": 0, "stopped": "max_steps"},
              trace_id=trace_id)
    return "Agent stopped: too many reasoning steps."
