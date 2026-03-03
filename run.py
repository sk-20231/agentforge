#!/usr/bin/env python
"""
Thin entry point for the agent CLI.
Run from the project root:  python run.py
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
            result = run_agent(user_id, session_id, user_input, history=history)
            log_event("final_answer", {"text": result})
            print("Agent:", result)

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result})
        except Exception as e:
            print("Error:", e)
