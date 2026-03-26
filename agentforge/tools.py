# agent/tools.py
import ast
import json
import logging
import re
from openai import OpenAI
from typing import Dict, Callable, Any, Union

from agentforge.config import OPENAI_MODEL, OPENAI_BASE_URL
from agentforge.prompts import SYSTEM_PROMPT, MEMORY_INSTRUCTIONS, OUTPUT_SCHEMA
from agentforge.memory.semantic import get_relevant_memories
from agentforge.logger import log_event, log_token_usage

logger = logging.getLogger(__name__)

_client = None  # created on first API call, not at import time

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()
    return _client

# -------------------- SAFE MATH EVAL --------------------

_ALLOWED_NAMES = frozenset({"abs", "round", "min", "max"})

def _safe_eval_node(node: ast.AST) -> Union[int, float]:
    """Evaluate an AST node that is restricted to numbers and safe math ops."""
    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, (int, float)):
            return val
        raise ValueError(f"Only numbers allowed, got {type(val).__name__}")
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_safe_eval_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return _safe_eval_node(node.operand)
        raise ValueError("Only + and - unary ops allowed")
    if isinstance(node, ast.BinOp):
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("Division by zero")
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ValueError("Division by zero")
            return left // right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ValueError("Modulo by zero")
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise ValueError("Unsupported operator")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed")
        if node.func.id not in _ALLOWED_NAMES:
            raise ValueError(f"Function not allowed: {node.func.id}")
        args = [_safe_eval_node(a) for a in node.args]
        if node.func.id == "abs" and len(args) == 1:
            return abs(args[0])
        if node.func.id == "round" and len(args) in (1, 2):
            return round(args[0]) if len(args) == 1 else round(args[0], int(args[1]))
        if node.func.id == "min" and args:
            return min(args)
        if node.func.id == "max" and args:
            return max(args)
        raise ValueError(f"Invalid call to {node.func.id}")
    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def _strip_leading_zeros(expression: str) -> str:
    """Remove leading zeros from integer literals so ast.parse doesn't reject them.

    Python 3 treats '0765' as invalid syntax (Python 2 octal). Users typing
    '0765 * 5243' mean 765, not octal. This regex finds sequences of digits
    with leading zeros and normalises them (e.g. '0765' → '765', '007' → '7'),
    while preserving '0' itself and decimal numbers like '0.5'.
    """
    return re.sub(r'\b0+(\d+)', r'\1', expression)


def safe_eval_math(expression: str) -> Union[int, float]:
    """
    Safely evaluate a math expression. Allows numbers, +, -, *, /, //, %, **,
    parentheses, and abs(), round(), min(), max(). No builtins or attribute access.
    """
    expr = _strip_leading_zeros(expression.strip())
    if not expr:
        raise ValueError("Empty expression")
    tree = ast.parse(expr, mode="eval")
    body = tree.body
    if not isinstance(body, (ast.BinOp, ast.UnaryOp, ast.Constant, ast.Call)):
        raise ValueError("Invalid expression form")
    return _safe_eval_node(body)


# -------------------- TOOLS --------------------

def calculator(expression: str) -> str:
    """Safely evaluate a math expression (no eval of arbitrary code)."""
    logger.info("Calculator tool invoked with expression: %s", expression)
    try:
        if not expression or not isinstance(expression, str):
            return "Error: expression must be a non-empty string"
        result = safe_eval_math(expression)
        return str(result)
    except ValueError as e:
        return f"Error calculating expression: {e}"
    except SyntaxError as e:
        return f"Error: invalid syntax in expression"


TOOL_REGISTRY: Dict[str, Callable] = {
    "calculator": calculator
}


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }
]

# -------------------- PROMPT BUILDER --------------------

def build_messages(user_id: str, user_input: str):
    memories = get_relevant_memories(user_id, user_input)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": f"{MEMORY_INSTRUCTIONS}\n\nRelevant memories:\n{memories}"
        },
        {"role": "user", "content": user_input},
        {"role": "system", "content": OUTPUT_SCHEMA}
    ]

# -------------------- TOOL EXECUTION --------------------

def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    log_event("tool_call", {
    "tool": name,
    "arguments": arguments
    })

    if name not in TOOL_REGISTRY:
        return f"Error: Unknown tool '{name}'"

    try:
        result =  TOOL_REGISTRY[name](**arguments)
        log_event("tool_result", {
        "tool": name,
        "result": result
        })
        return result
    except Exception as e:
        return f"Tool execution error: {e}"

# -------------------- MAIN ENTRY --------------------

def run_llm_with_tools(user_id: str, user_input: str, trace_id: str = None) -> str:
    """
    Executes the LLM call, handles tool calls if any,
    and returns the FINAL model output as a raw JSON string.
    """
    try:
        response = _get_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=build_messages(user_id, user_input),
            tools=TOOLS_SCHEMA,
            # "required" forces the model to call a tool even for trivially simple
            # expressions like 4*3 that it could compute mentally. Without this,
            # the model skips the tool, returns JSON with action.type="tool" and
            # reply="" (because OUTPUT_SCHEMA says reply must be empty when type
            # is tool), and the agent shows a blank response.
            tool_choice="required",
        )
        log_token_usage(response, "act_tool_call", trace_id=trace_id)
    except Exception as e:
        return json.dumps({
            "reply": "I couldn't complete that request due to a service error. Please try again.",
            "store_memory": False,
            "memory_text": ""
        })

    message = response.choices[0].message

    # ---------- TOOL CALL ----------
    if message.tool_calls:
        # Build tool response messages for ALL tool calls
        tool_messages = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "Error: Tool arguments were invalid."
                })
                continue

            tool_result = execute_tool(tool_name, arguments)
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        try:
            followup = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    *build_messages(user_id, user_input),
                    {
                        "role": "assistant",
                        "tool_calls": message.tool_calls
                    },
                    *tool_messages,
                    # Explicit finalise instruction: the model now has the tool
                    # result and must produce a final JSON reply, not another
                    # tool call. Without this, it sometimes returns reply: ""
                    # thinking another step is needed.
                    {
                        "role": "system",
                        "content": (
                            "You have received the tool result above. "
                            "Now produce your FINAL response. "
                            "Set action.type to 'final', write the answer to the user "
                            "in the 'reply' field, and do NOT call any more tools."
                        ),
                    },
                ]
            )
            log_token_usage(followup, "act_tool_followup", trace_id=trace_id)
            return followup.choices[0].message.content
        except Exception:
            return json.dumps({
                "reply": "I ran into an error after using the tool. Please try again.",
                "store_memory": False,
                "memory_text": ""
            })

    # ---------- NO TOOL ----------
    return response.choices[0].message.content
