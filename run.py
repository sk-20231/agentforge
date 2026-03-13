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
from agentforge.main import run_agent
from agentforge.logger import log_event


if __name__ == "__main__":
    print("🤖 AI Agent (type 'exit' to quit)\n")
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
            result = run_agent(user_id, session_id, user_input, history=history, stream=True)

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
