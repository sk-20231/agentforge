"""HackerNews top-stories tool.

Uses the Algolia HackerNews Search API to find recent high-quality
stories on a topic. Output is sanitized and wrapped in untrusted-data
delimiters so the LLM treats story titles as data, not instruction.
"""
import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from agentforge.tools._safety import extract_domain, sanitize_text

logger = logging.getLogger(__name__)

# search_by_date sorts newest first; the numericFilter keeps out brand-new
# low-quality posts. Result: recent stories with at least some traction.
HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
DEFAULT_MIN_POINTS = 20
DEFAULT_HITS_PER_PAGE = 5


def get_top_news(topic: str) -> str:
    """Return top HN stories for a topic as a compact, sanitized list.

    Returns raw, *sanitized* text on success; RAISES (``ValueError``) on failure
    so the MCP server reports ``isError: true``. It no longer wraps the result —
    the MCP gateway wraps tool output as untrusted data with the turn's nonce
    (Step 17e). Per-title sanitization (titles are user-submitted) stays here.
    """
    logger.info("HN news lookup invoked for topic: %s", topic)
    if not topic or not isinstance(topic, str):
        raise ValueError("topic must be a non-empty string")

    try:
        params = urllib.parse.urlencode({
            "query": topic.strip(),
            "tags": "story",
            "numericFilters": f"points>{DEFAULT_MIN_POINTS}",
            "hitsPerPage": DEFAULT_HITS_PER_PAGE,
        })
        req = urllib.request.Request(
            f"{HN_SEARCH_URL}?{params}",
            headers={"User-Agent": "AgentForge/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        hits = data.get("hits") or []
        if not hits:
            raise ValueError(f"No recent HN stories found for '{topic}'")

        clean_topic = sanitize_text(topic, 60)
        lines = [f"Top HN stories for '{clean_topic}':"]
        for i, hit in enumerate(hits, start=1):
            title = sanitize_text(hit.get("title") or "", 200)
            if not title:
                continue
            points = hit.get("points") or 0
            domain = extract_domain(hit.get("url") or "")
            domain_suffix = f" — {domain}" if domain else ""
            lines.append(f"{i}. {title} ({points} pts){domain_suffix}")

        return "\n".join(lines)

    except urllib.error.HTTPError as e:
        raise ValueError(f"HN API error (HTTP {e.code}) for '{topic}'") from e
    except urllib.error.URLError as e:
        raise ValueError(f"Could not reach HN search: {e.reason}") from e


TOOL_FUNCTION = get_top_news

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_top_news",
        "description": (
            "Search HackerNews for recent top stories on a topic. "
            "Returns up to 5 stories with titles, points, and source domains. "
            "Use for tech news and developer-focused topics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Keyword or topic to search for, e.g. 'OpenAI', 'Rust', 'quantum computing'",
                }
            },
            "required": ["topic"],
        },
    },
}
