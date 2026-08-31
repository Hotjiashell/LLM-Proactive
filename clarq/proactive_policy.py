"""LLM-Proactive decision policy adapted to ClarQ's tool protocol.

The Huawei evaluator owns the agent loop and calls ``policy_chat`` once per
turn. This adapter asks a regular OpenAI-compatible model for one structured
Proactive decision, validates it locally, and exposes the selected action as a
standard OpenAI tool call understood by that evaluator.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Mapping, Sequence

from clarq_eval.clients import response_content
from clarq_eval.parsing import PolicyProtocolError, parse_json_object


POLICY_NAME = "llm-proactive-clarq"
POLICY_VERSION = "1.0"
PROMPT_VERSION = "2026-08-31"


PROACTIVE_SYSTEM_PROMPT = """You are the decision component of a conversational case-retrieval agent.

Use the LLM-Proactive policy pattern: before selecting an action, make a short
explicit decision about whether a single missing user-known fact is needed,
whether the request is ready for retrieval, whether retrieved cases need a
better query, or whether the latest retrieval is sufficient.

You may take exactly one action:
- clarify_user: ask one concise, discriminative question only when its answer
  can materially change which cases are relevant. Do not ask for information
  already answered in the trace.
- search_case: retrieve cases using a focused query grounded only in the
  original request and confirmed user replies.
- Complete: choose only after at least one search, when the latest retrieved
  cases are sufficient. Do not answer the technical question yourself.

The trace is the only state you may use. It deliberately excludes the hidden
user profile, target case, and reference answer.

Return exactly one JSON object and no Markdown or surrounding prose:
{
  "decision": {
    "state": "needs_clarification | ready_to_search | refine_search | complete",
    "missing_information": "a short missing fact, or none",
    "basis": "a short factual decision summary, not chain-of-thought"
  },
  "action": {
    "name": "clarify_user | search_case | Complete",
    "arguments": {"question": "..."}
  }
}

For search_case use {"query": "..."} as arguments. For Complete use an empty
object as arguments. Do not emit tool markup; the caller converts this JSON to
the evaluator's native tool protocol."""


def _text(value: Any, *, default: str = "", limit: int = 500) -> str:
    if not isinstance(value, str):
        return default
    return " ".join(value.split())[:limit]


def _tool_summary(raw_tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tool_calls, list):
        return []
    calls: list[dict[str, Any]] = []
    for raw_call in raw_tool_calls:
        if not isinstance(raw_call, Mapping):
            continue
        function = raw_call.get("function")
        if not isinstance(function, Mapping):
            continue
        name = _text(function.get("name"), limit=80)
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        calls.append({"name": name, "arguments": arguments})
    return calls


def _trace_from_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Make a model-visible state trace without evaluator-only profile fields."""

    trace: list[dict[str, Any]] = []
    for message in messages:
        role = _text(message.get("role"), limit=40)
        if role == "system":
            # The runner's system prompt is restated by PROACTIVE_SYSTEM_PROMPT.
            continue
        entry: dict[str, Any] = {"role": role or "unknown"}
        content = message.get("content")
        if content is not None:
            entry["content"] = str(content)
        tool_calls = _tool_summary(message.get("tool_calls"))
        if tool_calls:
            entry["tool_calls"] = tool_calls
        tool_name = _text(message.get("name"), limit=80)
        if tool_name:
            entry["tool_name"] = tool_name
        trace.append(entry)
    return trace


def _normalise_action(payload: Mapping[str, Any]) -> tuple[str, dict[str, str], dict[str, str]]:
    """Validate one decision object and return its native ClarQ action."""

    raw_decision = payload.get("decision")
    decision = raw_decision if isinstance(raw_decision, Mapping) else {}
    action_value = payload.get("action")
    if isinstance(action_value, Mapping):
        raw_name = action_value.get("name")
        raw_arguments = action_value.get("arguments", payload.get("arguments", {}))
    else:
        raw_name = action_value
        raw_arguments = payload.get("arguments", {})

    name = _text(raw_name, limit=80)
    if not name:
        raise PolicyProtocolError("LLM-Proactive response is missing action.name")
    if not isinstance(raw_arguments, Mapping):
        raise PolicyProtocolError("LLM-Proactive action.arguments must be a JSON object")

    reported_decision = {
        "state": _text(decision.get("state"), default="unspecified", limit=120),
        "missing_information": _text(decision.get("missing_information"), default="none", limit=500),
        "basis": _text(decision.get("basis"), default="", limit=500),
    }

    if name.lower() == "complete":
        if raw_arguments:
            raise PolicyProtocolError("Complete must use empty action.arguments")
        return "Complete", {}, reported_decision
    if name == "clarify_user":
        question = _text(raw_arguments.get("question"), limit=1000)
        if not question:
            raise PolicyProtocolError("clarify_user requires a non-empty arguments.question")
        return name, {"question": question}, reported_decision
    if name == "search_case":
        query = _text(raw_arguments.get("query"), limit=2000)
        if not query:
            raise PolicyProtocolError("search_case requires a non-empty arguments.query")
        return name, {"query": query}, reported_decision
    raise PolicyProtocolError(f"Unsupported LLM-Proactive action: {name!r}")


class ProactivePolicyClient:
    """Convert a structured LLM-Proactive decision into a ClarQ tool response."""

    def __init__(self, client: Any):
        self.client = client
        self._call_lock = threading.Lock()
        self._next_call_number = 0
        self._local = threading.local()

    def begin_sample(self, sample: Any) -> None:
        self._local.sample_id = str(getattr(sample, "sample_id", ""))
        self._local.decisions = []

    def finish_sample(self) -> list[dict[str, Any]]:
        decisions = getattr(self._local, "decisions", [])
        self._local.decisions = []
        return [dict(item) for item in decisions]

    def policy_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        model_mode: str = "qwen3_5",
        tokenizer_path: str | None = None,
        enable_thinking: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if enable_thinking:
            raise ValueError("LLM-Proactive ClarQ requires --no-policy-enable-thinking for JSON protocol stability")
        if not tools:
            raise ValueError("LLM-Proactive ClarQ requires the evaluator's action tool definitions")

        trace = _trace_from_messages(messages)
        prompt_messages = [
            {"role": "system", "content": PROACTIVE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Conversation and tool trace:\n" + json.dumps(trace, ensure_ascii=False),
            },
        ]
        response = self._request_decision(
            prompt_messages,
            model_mode=model_mode,
            tokenizer_path=tokenizer_path,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        payload = parse_json_object(response_content(response))
        name, arguments, reported_decision = _normalise_action(payload)
        self._record_decision(reported_decision, name, arguments)

        if name == "Complete":
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Complete", "tool_calls": []},
                    }
                ]
            }

        with self._call_lock:
            self._next_call_number += 1
            call_id = f"proactive_call_{self._next_call_number}"
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": arguments},
                            }
                        ],
                    },
                }
            ]
        }

    def _request_decision(
        self,
        messages: list[dict[str, Any]],
        *,
        model_mode: str,
        tokenizer_path: str | None,
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> dict[str, Any]:
        if model_mode == "qwen3_5":
            return self.client.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
            )
        if model_mode == "qwen3":
            # Reuse Huawei's local tokenizer rendering and disabled-thinking protocol.
            return self.client.user_simulator_chat(
                messages,
                model_mode=model_mode,
                tokenizer_path=tokenizer_path,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
        raise ValueError(f"Unsupported policy model_mode: {model_mode!r}")

    def _record_decision(self, decision: dict[str, str], name: str, arguments: dict[str, str]) -> None:
        decisions = getattr(self._local, "decisions", None)
        if decisions is None:
            decisions = []
            self._local.decisions = decisions
        decisions.append(
            {
                "turn": len(decisions) + 1,
                "decision": decision,
                "action": {"name": name, "arguments": dict(arguments)},
            }
        )
