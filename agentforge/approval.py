"""Human-in-the-loop approval contract for high-impact MCP tool calls (Step 17f).

WHY this exists (the AI-engineering concept):
    The gateway's automated defenses (Step 17e) catch known-bad patterns —
    shadowed tools, rug-pulled definitions, private URLs. But a call that looks
    legitimate to every automated check can still be harmful: a poisoned headline
    can steer the model into fetching an attacker's *public* URL with sensitive
    context encoded in the query string. No rule distinguishes that from a
    legitimate fetch — only a human can. Per Meta's "Agents Rule of Two", a
    session that [A] processes untrusted input, [B] holds private data, and
    [C] can communicate externally needs a human decision on the dangerous leg.
    Our sessions always have [A] and [B], so calls adding [C] (open-world tools
    like ``fetch``) are gated behind human confirmation.

HOW the contract works (dependency inversion):
    The gateway (``agentforge.mcp_client``) owns the *decision point* — it knows
    which call needs approval — but only the front-end knows *how* to reach the
    human. So the gateway accepts an ``approval_handler`` callable and stays
    UI-ignorant. A handler receives an :class:`ApprovalRequest` and either:

    - returns ``True``  → approved, the call dispatches;
    - returns ``False`` → denied, the model receives a readable
      "user declined" observation it can adapt to (deny ≠ crash); or
    - raises :class:`ApprovalRequired` → "I cannot answer synchronously."

    No handler configured (tests, eval runs, future callers) → the gateway
    DENIES by default. Nothing ever silently auto-approves a gated call.

INTERRUPT → CHECKPOINT → RESUME (the non-blocking front-end story):
    Streamlit cannot pause mid-run to ask a question, so its handler raises
    :class:`ApprovalRequired`. As the exception unwinds, the pipeline loop
    attaches a **continuation** to it — the pending call plus the frozen loop
    state (the messages list, the step index). The front-end stashes the
    exception, renders Allow/Deny, and on click calls the pipeline's resume
    entry point, which re-enters the loop mid-flight and settles the *stored*
    call with the human's decision via :func:`make_resume_handler`.

    Resume — not replay — is deliberate: a human approved a SPECIFIC action.
    Re-running the turn would ask the non-deterministic LLM to regenerate that
    action and hope it matches (it often doesn't — argument values drift between
    generations). Binding the decision to the stored action removes the hope.
    This is hand-rolled LangGraph ``interrupt()`` + checkpointer (the production
    equivalent, planned for the Step 21 multi-agent track).
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class ApprovalRequest:
    """One pending tool call awaiting a human decision.

    Carries everything a person needs to make an informed call: WHICH tool,
    from WHICH server, with WHAT arguments. Arguments may contain user data —
    they are shown to the user (it is their own data and their decision) but
    must never be written to logs (the audit trail records arg *names* only,
    same rule as Step 17e gap A).
    """
    tool: str
    server: str
    arguments: Dict[str, Any] = field(default_factory=dict)


class ApprovalRequired(Exception):
    """Raised by an approval handler that cannot answer synchronously.

    Deliberately unwinds the whole turn (gateway → loop → run_agent → front-end).
    On the way out, the pipeline loop sets ``continuation`` — the checkpoint a
    resume entry point needs to re-enter the loop mid-flight (pipeline tag,
    frozen messages, step index, the pending call). The front-end catches the
    exception, presents ``request`` to the human, and calls
    ``main.resume_agent(exc, decision)``. It must NOT be swallowed anywhere
    along the way — a swallowed interrupt would silently drop the user's turn.
    """

    def __init__(self, request: ApprovalRequest):
        self.request = request
        # Set by the pipeline loop as the exception unwinds (the loop, not the
        # gateway, owns the loop state). None means "not resumable" — e.g. a
        # handler raised outside any pipeline.
        self.continuation: Optional[dict] = None
        super().__init__(
            f"Human approval required for tool '{request.tool}' (server '{request.server}')"
        )


# The contract front-ends implement: ApprovalRequest in → decision out (or raise).
ApprovalHandler = Callable[[ApprovalRequest], bool]


def make_resume_handler(decision: bool, pending: ApprovalRequest,
                        fallback: Optional[ApprovalHandler] = None) -> ApprovalHandler:
    """Build the approval handler for a resumed turn: a one-shot grant.

    The human's Allow/Deny applies to **exactly the stored pending call** —
    same tool, same server, same arguments — and is consumed on first use.
    Anything else the resumed loop tries goes to ``fallback`` (the front-end's
    normal handler, so a NEW gated call triggers a fresh interrupt/card), or is
    denied when there is no fallback. This keeps the security property intact:
    a "yes" can never bless a call the human didn't look at.
    """
    state = {"used": False}

    def handler(request: ApprovalRequest) -> bool:
        if (
            not state["used"]
            and request.tool == pending.tool
            and request.server == pending.server
            and request.arguments == pending.arguments
        ):
            state["used"] = True
            return decision
        if fallback is not None:
            return fallback(request)
        return False  # fail-safe: no way to ask -> deny

    return handler


# Python 3.11+ has BaseExceptionGroup as a builtin; on 3.10 anyio guarantees the
# exceptiongroup backport is installed (it's a dependency of anyio itself).
try:
    _BaseExceptionGroup = BaseExceptionGroup
except NameError:  # pragma: no cover - Python 3.10 only
    from exceptiongroup import BaseExceptionGroup as _BaseExceptionGroup


def unwrap_approval_required(exc: BaseException):
    """Return the :class:`ApprovalRequired` inside ``exc``, or ``None``.

    The MCP SDK's stdio transport runs inside anyio task groups. When the
    approval handler raises ApprovalRequired and it unwinds through those task
    groups, Python wraps it into an ``ExceptionGroup`` ("unhandled errors in a
    TaskGroup") — so the front-end's ``except ApprovalRequired`` would never
    match. :func:`run_interruptible` calls this to dig the real interrupt out
    of the (possibly nested) group and re-raise it bare, keeping the documented
    contract: callers above the bridge always see ApprovalRequired itself.
    Found live in Streamlit testing; the mocked-transport tests couldn't
    reproduce it (no real task groups — the "mocks at the wrong layer" lesson
    again).
    """
    if isinstance(exc, ApprovalRequired):
        return exc
    if isinstance(exc, _BaseExceptionGroup):
        for sub in exc.exceptions:
            found = unwrap_approval_required(sub)
            if found is not None:
                return found
    return None


def run_interruptible(coro):
    """``asyncio.run(coro)``, re-raising a task-group-wrapped interrupt bare.

    Every sync→async bridge that can carry an approval interrupt
    (``run_llm_with_tools`` / ``react_loop`` and their resume twins) runs its
    coroutine through this so callers above the bridge always see the bare
    ApprovalRequired (with its ``continuation`` intact — the same exception
    object is re-raised, never a copy). Unrelated exceptions and groups
    propagate unchanged.
    """
    try:
        return asyncio.run(coro)
    except BaseException as exc:
        interrupt = unwrap_approval_required(exc)
        if interrupt is not None and interrupt is not exc:
            raise interrupt from None
        raise
