"""One-call ProCoT-style policy using Huawei ClarQ's native tools.

The Huawei evaluator owns the agent loop and calls ``policy_chat`` once per
turn. The policy model first writes an unconstrained natural-language analysis,
then selects one of Huawei's native tools in that same response. This adapter
records the analysis and normalizes the selected action for the evaluator.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Mapping, Sequence

from clarq_eval.clients import response_content
from clarq_eval.parsing import PolicyProtocolError, parse_policy_response


POLICY_NAME = "llm-proactive-clarq"
POLICY_VERSION = "3.0"
PROMPT_VERSION = "2026-08-31-procot-native-tools"


PROACTIVE_SYSTEM_PROMPT = """You are the decision component of a conversational case-retrieval agent.

Use this analysis pattern: First analyze the current conversation state in ordinary natural language. Consider whether a single missing user-known fact is needed, whether the request is ready for retrieval, whether retrieved cases need a better query, or whether the latest retrieval is sufficient.

You may take exactly one action:
- clarify_user: ask one concise, discriminative question only when its answer can materially change which cases are relevant. Do not ask for information already answered in the trace. The language used must be consistent with the user’s language; for example, if the user speaks Chinese, you should also use Chinese.
- search_case: retrieve cases using a focused query grounded only in the original request and confirmed user replies. The language used must be consistent with the user’s language; for example, if the user speaks Chinese, you should also use Chinese.
- Complete: choose only after at least one search, when the latest retrieved cases are sufficient. Do not answer the technical question yourself.

The trace is the only state you may use.

For clarify_user or search_case, put the analysis in the assistant content and then make exactly one native tool call using the tool definitions supplied with this request. Do not write a tool-call JSON object in the assistant content.

For Complete, do not make a tool call. Put the analysis first, then make the last non-empty line exactly `Complete`. If there is no useful analysis, output only `Complete`.
"""


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


def _normalise_action(payload: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    """Validate the terminal structured action and return its ClarQ form."""

    raw_name = payload.get("name")
    raw_arguments = payload.get("arguments", {})

    name = _text(raw_name, limit=80)
    if not name:
        raise PolicyProtocolError("LLM-Proactive response is missing action.name")
    if not isinstance(raw_arguments, Mapping):
        raise PolicyProtocolError("LLM-Proactive action.arguments must be a JSON object")

    if name.lower() == "complete":
        if raw_arguments:
            raise PolicyProtocolError("Complete must use empty action.arguments")
        return "Complete", {}
    if name == "clarify_user":
        question = _text(raw_arguments.get("question"), limit=1000)
        if not question:
            raise PolicyProtocolError("clarify_user requires a non-empty arguments.question")
        return name, {"question": question}
    if name == "search_case":
        query = _text(raw_arguments.get("query"), limit=2000)
        if not query:
            raise PolicyProtocolError("search_case requires a non-empty arguments.query")
        return name, {"query": query}
    raise PolicyProtocolError(f"Unsupported LLM-Proactive action: {name!r}")


def _normalized_model_response(response: dict[str, Any]) -> dict[str, Any]:
    """Give the shared parser a message-shaped response for both Qwen modes."""

    try:
        choice = response["choices"][0]
    except (KeyError, IndexError, TypeError) as error:
        raise PolicyProtocolError("Policy model response has no choices[0]") from error
    if not isinstance(choice, Mapping):
        raise PolicyProtocolError("Policy model response choices[0] must be an object")
    if isinstance(choice.get("message"), Mapping):
        return response
    return {
        "choices": [
            {
                "finish_reason": choice.get("finish_reason"),
                "message": {"content": response_content(response), "tool_calls": []},
            }
        ]
    }


def _analysis_before_complete(content: str) -> str:
    """Extract optional free-form analysis from a terminal ProCoT response."""

    lines = content.strip().splitlines()
    if not lines or lines[-1].strip().lower() != "complete":
        raise PolicyProtocolError(
            "LLM-Proactive terminal response must end with a line containing exactly Complete"
        )
    return "\n".join(lines[:-1]).strip()


class ProactivePolicyClient:
    """Convert a one-call ProCoT response into a ClarQ tool response."""

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
        response = self._request_response(
            prompt_messages,
            tools=tools,
            model_mode=model_mode,
            tokenizer_path=tokenizer_path,
            enable_thinking=enable_thinking,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        parsed = parse_policy_response(_normalized_model_response(response))
        if len(parsed.tool_calls) > 1:
            raise PolicyProtocolError("LLM-Proactive response must contain at most one native tool call")
        if parsed.tool_calls:
            call = parsed.tool_calls[0]
            analysis = parsed.cleaned_content
            name, arguments = _normalise_action({"name": call.name, "arguments": call.arguments})
        else:
            analysis = _analysis_before_complete(parsed.cleaned_content)
            name, arguments = "Complete", {}
        self._record_decision(analysis, name, arguments)

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

    def _request_response(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        model_mode: str,
        tokenizer_path: str | None,
        enable_thinking: bool,
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> dict[str, Any]:
        return self.client.policy_chat(
            messages,
            tools=tools,
            model_mode=model_mode,
            tokenizer_path=tokenizer_path,
            enable_thinking=enable_thinking,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

    def _record_decision(self, analysis: str, name: str, arguments: dict[str, str]) -> None:
        decisions = getattr(self._local, "decisions", None)
        if decisions is None:
            decisions = []
            self._local.decisions = decisions
        decisions.append(
            {
                "turn": len(decisions) + 1,
                "analysis": analysis,
                "action": {"name": name, "arguments": dict(arguments)},
            }
        )
