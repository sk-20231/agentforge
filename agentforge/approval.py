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

    A handler MAY expose ``resuming = True`` when its next answer is a
    decision the human already made (a resumed turn re-running the paused
    call). The gateway then audits ``approval_resumed`` instead of a second
    ``approval_requested``, so the trail shows one request per human decision.

INTERRUPT → CHECKPOINT → RESUME (the non-blocking front-end story):
    Streamlit cannot pause mid-run to ask a question, so its handler raises
    :class:`ApprovalRequired`. Both tool pipelines run as LangGraph graphs
    (Step 21.1, ``agentforge.hitl_graph``): the bridge handler turns that into
    ``interrupt()``, which pauses the graph and checkpoints the turn's state
    under a ``thread_id``. The pause reaches the front-end as ApprovalRequired
    whose ``continuation`` carries only that handle. The front-end renders
    Allow/Deny and on click calls ``main.resume_agent``, which resumes the
    graph with ``Command(resume=decision)``.

    Resume — not replay — is deliberate: a human approved a SPECIFIC action.
    Re-running the turn would ask the non-deterministic LLM to regenerate that
    action and hope it matches (it often doesn't — argument values drift between
    generations). LangGraph re-runs only the paused node, which reads the call
    from checkpointed state; the LLM calls live in nodes that never pause.
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

    Inside a pipeline the bridge handler catches it and pauses the graph
    instead; the pipeline then raises a fresh ApprovalRequired toward the
    front-end with ``continuation`` set to the handle a resume needs:
    ``{"pipeline", "thread_id", "user_id", "trace_id"}``. The turn's state
    itself lives in the checkpointer. The front-end catches it, presents
    ``request`` to the human, and calls ``main.resume_agent(exc, decision)``.
    It must NOT be swallowed anywhere along the way — a swallowed interrupt
    would silently drop the user's turn.
    """

    def __init__(self, request: ApprovalRequest):
        self.request = request
        # Set by the pipeline when its graph pauses. None means "not
        # resumable" — e.g. a handler raised outside any pipeline.
        self.continuation: Optional[dict] = None
        super().__init__(
            f"Human approval required for tool '{request.tool}' (server '{request.server}')"
        )


# A handler's decision is tri-state (issue #6):
#   False        → deny this call
#   True         → allow exactly this call (one-shot)
#   APPROVE_TURN → allow this call AND every later call to the SAME tool for
#                  the REST OF THIS TURN. The standard answer to approval
#                  fatigue (a paging fetch raised 5 cards for one article):
#                  asking five times trains the human to rubber-stamp, which
#                  is worse than asking once. Deliberately narrower than a
#                  persistent "always allow" — the grant lives in the turn's
#                  own state (the gateway's ``granted`` set, written to the
#                  graph's checkpointed state across pauses), so it expires
#                  WITH the turn by construction. There is no cleanup step to forget:
#                  scope binds to the stored turn, not to ambient session
#                  state — same artifact-binding principle as resume itself.
# A truthy string keeps backward compatibility: any consumer that only checks
# ``if decision`` treats it as a plain approve-once.
APPROVE_TURN = "approve_turn"

# The contract front-ends implement: ApprovalRequest in → decision out (or
# raise). The decision is False / True / APPROVE_TURN (see above).
ApprovalHandler = Callable[[ApprovalRequest], Any]


# Python 3.11+ has BaseExceptionGroup as a builtin; on 3.10 anyio guarantees the
# exceptiongroup backport is installed (it's a dependency of anyio itself).
try:
    _BaseExceptionGroup = BaseExceptionGroup
except NameError:  # pragma: no cover - Python 3.10 only
    from exceptiongroup import BaseExceptionGroup as _BaseExceptionGroup


def unwrap_approval_required(exc: BaseException):
    """Return the :class:`ApprovalRequired` inside ``exc``, or ``None``.

    The MCP SDK's stdio transport runs inside anyio task groups. When a
    paused pipeline raises ApprovalRequired inside the open gateway and it
    unwinds through those task groups, Python wraps it into an
    ``ExceptionGroup`` ("unhandled errors in a TaskGroup") — so the
    front-end's ``except ApprovalRequired`` would never match. :func:`run_interruptible` calls this to dig the real interrupt out
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
