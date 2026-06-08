"""Backward-compat shim.

The safety helpers moved to ``agentforge.safety`` in Step 17d so the MCP gateway
(``agentforge.mcp_client``) can reuse them without a circular import. The tool
modules still import from here, so this re-exports the public names. New code
should import from ``agentforge.safety`` directly.
"""
from agentforge.safety import (  # noqa: F401
    extract_domain,
    is_safe_url,
    sanitize_text,
    wrap_untrusted,
)
