# Security Policy

## Supported Versions

AgentForge is under active development. There are no tagged releases yet — the `main` branch is always the current, supported version, and security fixes are applied there directly.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

If you find a security issue, please reach out via **LinkedIn (Sayali Kulkarni)** with:
- A description of the vulnerability
- Steps to reproduce it
- Any suggested fix (optional but appreciated)

I'll respond within a few days, and I take every report seriously — confirmed issues are addressed promptly.

## Scope

The main areas worth reviewing for security issues:

- **Prompt injection defences** — tool results from external sources (Wikipedia, HackerNews, weather) are sanitized and wrapped before being passed to the LLM. See `agentforge/tools/_safety.py`.
- **API key handling** — the OpenAI API key is read from environment variables only. It is never logged or stored. The `.env` file is gitignored.
- **Corpus and memory files** — currently stored locally as JSON. The current single-user local deployment applies no additional authentication; per-user isolation and auth are on the roadmap as the project moves toward multi-user use.

## Out of Scope

- Attacks that require physical access to the machine running the agent
- Issues in third-party dependencies (report those upstream)
