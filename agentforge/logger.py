import json
from datetime import datetime

from agentforge.config import AGENT_LOG_FILE

def log_event(event_type: str, payload: dict):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event_type,
        "payload": payload
    }

    with open(AGENT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
