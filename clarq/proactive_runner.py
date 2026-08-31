"""Attach LLM-Proactive decisions to unmodified Huawei ClarQ trajectories."""

from __future__ import annotations

from typing import Any

from proactive_policy import POLICY_NAME, POLICY_VERSION, PROMPT_VERSION, ProactivePolicyClient


class ProactiveTraceRunner:
    """Delegate evaluation work while recording ProCoT analysis and final actions."""

    def __init__(self, runner: Any, policy_client: ProactivePolicyClient):
        self._runner = runner
        self._policy_client = policy_client

    @property
    def user_simulator(self) -> Any:
        return self._runner.user_simulator

    def run(self, sample: Any) -> dict[str, Any]:
        self._policy_client.begin_sample(sample)
        try:
            result = self._runner.run(sample)
        except Exception:
            # Huawei writes its own infrastructure-error record when run() raises.
            self._policy_client.finish_sample()
            raise
        result["proactive_policy"] = {
            "name": POLICY_NAME,
            "version": POLICY_VERSION,
            "prompt_version": PROMPT_VERSION,
            "decisions": self._policy_client.finish_sample(),
        }
        return result
