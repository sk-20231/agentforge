# Security Policy

## Supported Versions

This is a personal learning project. There are no versioned releases — the `main` branch reflects the current state of the project.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

If you find a security issue, please reach out via **LinkedIn (Sayali Kulkarni)** with:
- A description of the vulnerability
- Steps to reproduce it
- Any suggested fix (optional but appreciated)

I'll respond within a few days. For a learning project like this, I don't expect critical vulnerabilities, but I take the report seriously and will address it promptly.

## Scope

The main areas worth reviewing for security issues:

- **Prompt injection defences** — tool results from external sources (Wikipedia, HackerNews, weather) are sanitized and wrapped before being passed to the LLM. See `agentforge/tools/_safety.py`.
- **API key handling** — the OpenAI API key is read from environment variables only. It is never logged or stored. The `.env` file is gitignored.
- **Corpus and memory files** — stored locally as JSON. No authentication is applied since this is a single-user local agent.

## Out of Scope

- Attacks that require physical access to the machine running the agent
- Issues in third-party dependencies (report those upstream)
