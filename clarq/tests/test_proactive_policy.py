from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


CLARQ_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = CLARQ_DIR.parents[1] / "huawei_dial" / "workspace" / "eval"
for path in (CLARQ_DIR, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from clarq_eval.models import EvaluationSample  # noqa: E402
from clarq_eval.parsing import PolicyProtocolError, parse_policy_response  # noqa: E402
from clarq_eval.runner import EvaluationRunner, TOOLS  # noqa: E402
from proactive_policy import ProactivePolicyClient  # noqa: E402
from proactive_runner import ProactiveTraceRunner  # noqa: E402
from run_evaluation import DEFAULT_EVAL_DIR, _install_proactive_adapter, _load_evaluator  # noqa: E402


def model_response(
    content: str | None,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": content, "tool_calls": tool_calls or []},
            }
        ]
    }


def native_tool_call(name: str, arguments: dict[str, str]) -> dict[str, Any]:
    return {
        "id": "model_call_1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class FakePolicyService:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def policy_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("Policy response sequence exhausted")
        return self.responses.pop(0)


class FakeRetriever:
    def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        if "Model X" in query:
            return [{"case_id": "target", "title": "Target case", "content": "Target answer"}]
        return [{"case_id": "other", "title": "Other case", "content": "Other answer"}]


class FakeSimulator:
    def answer(self, sample: EvaluationSample, question: str) -> str:
        self.last_question = question
        return "Model X"


class FakeSuccessJudge:
    def judge(self, sample: EvaluationSample, cases: list[dict[str, Any]]) -> dict[str, Any]:
        raise AssertionError("The target title is in final results, so this Judge must not run")


class ProactivePolicyTests(unittest.TestCase):
    def test_freeform_analysis_ends_in_one_native_tool_call(self) -> None:
        service = FakePolicyService(
            [
                model_response(
                    "Different device models can lead to different support cases. "
                    "The requested model has not been confirmed.",
                    tool_calls=[
                        native_tool_call(
                            "clarify_user",
                            {"question": "Which device model are you using?"},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            ]
        )
        policy = ProactivePolicyClient(service)
        policy.begin_sample(type("Sample", (), {"sample_id": "sample-1"})())

        response = policy.policy_chat(
            [
                {"role": "system", "content": "Runner instruction"},
                {"role": "user", "content": "My device has an error."},
            ],
            tools=TOOLS,
            enable_thinking=True,
            temperature=0.0,
            max_tokens=256,
            seed=7,
        )

        turn = parse_policy_response(response)
        self.assertEqual("clarify_user", turn.tool_calls[0].name)
        self.assertEqual({"question": "Which device model are you using?"}, turn.tool_calls[0].arguments)
        self.assertEqual([], turn.violations)
        self.assertTrue(service.calls[0]["kwargs"]["enable_thinking"])
        self.assertIs(TOOLS, service.calls[0]["tools"])
        self.assertEqual("Complete", service.calls[0]["tools"][2]["function"]["name"])
        self.assertEqual(1, len(service.calls))
        self.assertNotIn("Runner instruction", service.calls[0]["messages"][1]["content"])
        decision = policy.finish_sample()[0]
        self.assertEqual(
            "Different device models can lead to different support cases. The requested model has not been confirmed.",
            decision["analysis"],
        )
        self.assertNotIn("missing_information", decision["analysis"])

    def test_invalid_action_is_rejected_before_the_agent_loop(self) -> None:
        service = FakePolicyService(
            [
                model_response(
                    "The request is ready for the next step.",
                    tool_calls=[native_tool_call("answer_user", {})],
                    finish_reason="tool_calls",
                )
            ]
        )
        policy = ProactivePolicyClient(service)

        with self.assertRaisesRegex(PolicyProtocolError, "Unsupported LLM-Proactive action"):
            policy.policy_chat(
                [{"role": "user", "content": "question"}],
                tools=TOOLS,
            )

    def test_complete_tool_without_analysis_is_valid(self) -> None:
        service = FakePolicyService(
            [
                model_response(
                    None,
                    tool_calls=[native_tool_call("Complete", {})],
                    finish_reason="tool_calls",
                )
            ]
        )
        policy = ProactivePolicyClient(service)

        response = policy.policy_chat(
            [{"role": "user", "content": "question"}],
            tools=TOOLS,
        )

        turn = parse_policy_response(response)
        self.assertEqual("Complete", turn.tool_calls[0].name)
        self.assertEqual({}, turn.tool_calls[0].arguments)
        self.assertEqual("", policy.finish_sample()[0]["analysis"])

    def test_trace_runner_uses_huawei_loop_and_preserves_decisions(self) -> None:
        service = FakePolicyService(
            [
                model_response(
                    "The original request does not identify the model, which changes retrieval.",
                    tool_calls=[
                        native_tool_call(
                            "clarify_user",
                            {"question": "Which device model are you using?"},
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                model_response(
                    "The user has now confirmed Model X, so the query can be specific.",
                    tool_calls=[native_tool_call("search_case", {"query": "device error Model X"})],
                    finish_reason="tool_calls",
                ),
                model_response(
                    "The latest retrieved case is a sufficient match for the confirmed model.",
                    tool_calls=[native_tool_call("Complete", {})],
                    finish_reason="tool_calls",
                ),
            ]
        )
        policy = ProactivePolicyClient(service)
        core_runner = EvaluationRunner(
            policy_client=policy,
            user_simulator=FakeSimulator(),
            retriever=FakeRetriever(),
            success_judge=FakeSuccessJudge(),
            max_turns=4,
            max_searches=2,
            top_k=1,
            success_top_k=1,
        )
        runner = ProactiveTraceRunner(core_runner, policy)
        sample = EvaluationSample(
            sample_id="sample-1",
            domain="electronics",
            target_case_id="target",
            initial_question="My device has an error.",
            core_intent="hidden intent",
            known_info=("The device model is Model X.",),
            target_case_title="Target case",
            target_case_content="Target answer",
        )

        result = runner.run(sample)

        self.assertEqual(
            ["clarify_user", "search_case", "complete"],
            [event["action"]["type"] for event in result["events"]],
        )
        self.assertTrue(result["success_judgment"]["success"])
        decisions = result["proactive_policy"]["decisions"]
        self.assertEqual(3, len(decisions))
        self.assertEqual("clarify_user", decisions[0]["action"]["name"])
        self.assertEqual("search_case", decisions[1]["action"]["name"])
        self.assertEqual("Complete", decisions[2]["action"]["name"])
        self.assertIn("does not identify the model", decisions[0]["analysis"])
        second_prompt = service.calls[1]["messages"][1]["content"]
        self.assertIn("Model X", second_prompt)
        self.assertNotIn("hidden intent", second_prompt)

    def test_launcher_records_adapter_and_prevents_mixed_resume(self) -> None:
        evaluator = _load_evaluator(EVAL_DIR)
        _install_proactive_adapter(evaluator)
        with tempfile.TemporaryDirectory() as directory:
            args = evaluator.parse_args(
                [
                    "--output-dir",
                    directory,
                    "--policy-base-url",
                    "http://policy.example/v1",
                    "--policy-model",
                    "policy-model",
                    "--simulator-base-url",
                    "http://simulator.example/v1",
                    "--simulator-model",
                    "simulator-model",
                    "--skip-judge",
                    "--policy-enable-thinking",
                ]
            )
        config = evaluator._run_config(args)
        self.assertEqual("llm-proactive-clarq", config["policy_adapter"]["name"])
        self.assertTrue(config["policy_adapter"]["thinking_enabled"])
        self.assertEqual(
            "freeform_procot_analysis_with_huawei_native_tools",
            config["policy_adapter"]["protocol"],
        )
        native_config = dict(config)
        native_config.pop("policy_adapter")
        self.assertNotEqual(evaluator._resume_signature(config), evaluator._resume_signature(native_config))

    def test_default_evaluator_path_points_to_sibling_huawei_checkout(self) -> None:
        self.assertEqual(EVAL_DIR.resolve(), DEFAULT_EVAL_DIR.resolve())


if __name__ == "__main__":
    unittest.main()
