#!/usr/bin/env python
"""
Thin entry point for the agent CLI.
Run from the project root:  python run.py

Streaming behaviour (Step 4):
  run_agent returns either a str or an Iterator[str] depending on intent.
  - ANSWER / RESPOND_WITH_MEMORY → Iterator[str]: tokens printed as they arrive.
  - All other intents (REMEMBER, ACT, REACT, DOCS_QA, IGNORE) → str: printed at once.

  The isinstance(result, str) check is the dispatch point. We collect the full
  text from the generator so it can be stored in history and logged, just like
  the non-streaming paths.
"""
import json

from agentforge.approval import APPROVE_TURN, ApprovalRequest
from agentforge.main import run_agent
from agentforge.logger import log_event
from agentforge.tools import prime_tool_catalog


def cli_approval_handler(request: ApprovalRequest):
    """Human-in-the-loop gate for the CLI (Step 17f; turn grants — issue #6).

    The terminal CAN block mid-turn, so this is the simple half of the approval
    contract: print what the agent wants to do, read the answer, return the
    decision: y = allow once, t = allow this tool for the rest of this turn
    (the fix for approval fatigue when e.g. a fetch pages one article in
    chunks), anything else = deny.
    The arguments are shown in full — the human can only make an informed
    decision if they can see exactly what would be sent (e.g. the URL a fetch
    would request). Shown on screen only; the audit log records arg names, never
    values. Default is DENY: anything but an explicit y/t declines the call.

    (Known limit, documented in mcp_client: input() blocks the event loop —
    fine for a single-user CLI, revisit for the multi-user product.)
    """
    print("\n⚠️  The agent wants to call a gated tool:")
    print(f"    tool:      {request.tool}")
    print(f"    server:    {request.server}  (requires approval)")
    print(f"    arguments: {json.dumps(request.arguments, ensure_ascii=False)}")
    answer = input("    Allow this call? [y = once / t = rest of this turn / N] ").strip().lower()
    if answer in ("t", "turn"):
        return APPROVE_TURN
    return answer in ("y", "yes")


if __name__ == "__main__":
    print("🤖 AI Agent (type 'exit' to quit)\n")
    # Discover the MCP tool catalog once at startup (Step 17c.1) so the intent
    # classifier knows the available tools and the cost is paid up front.
    prime_tool_catalog()
    user_id = input("Enter user id: ").strip()
    session_id = "default"
    history: list[dict] = []

    while True:
        user_input = input(f"{user_id}> ")
        if user_input.lower() == "exit":
            break

        try:
            log_event("user_input", {"text": user_input})

            # stream=True: ANSWER/RESPOND_WITH_MEMORY intents return a generator.
            # All other intents still return a plain str — no change for them.
            # approval_handler: gated tool calls (Step 17f) pause here for a y/n.
            result = run_agent(user_id, session_id, user_input, history=history,
                               stream=True, approval_handler=cli_approval_handler)

            if isinstance(result, str):
                # Non-streaming intents (REMEMBER, ACT, REACT, DOCS_QA, IGNORE).
                log_event("final_answer", {"text": result, "streamed": False})
                print("Agent:", result)
            else:
                # Streaming intent (ANSWER / RESPOND_WITH_MEMORY).
                # Collect tokens into full_text while printing each one immediately.
                # `end=""` + `flush=True` ensures the token appears on screen
                # before the next one arrives — without flushing, Python buffers
                # stdout and the streaming effect is lost.
                print("Agent: ", end="", flush=True)
                full_text = ""
                for token in result:
                    print(token, end="", flush=True)
                    full_text += token
                print()  # newline after the streamed response
                result = full_text
                log_event("final_answer", {"text": result, "streamed": True})

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result})

        except Exception as e:
            print("Error:", e)
