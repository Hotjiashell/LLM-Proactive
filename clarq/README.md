# LLM-Proactive ClarQ

This directory applies LLM-Proactive's explicit intermediate-decision method
to the Huawei ClarQ conversational case-retrieval task. It is a thin adapter
over `../../huawei_dial/workspace/eval`, not a second evaluator.

The policy sees the initial request, earlier actions, user-simulator replies,
and retrieved cases. It never receives ClarQ's `core_intent`, `known_info`,
target case ID, target title, or reference answer. Each turn has one short
structured decision and one action. The adapter validates that decision, then
converts it to Huawei's standard `clarify_user` or `search_case` tool call, or
the `Complete` terminal response.

## Setup

Install Huawei evaluator dependencies in the Python environment used to run
this integration:

```bash
python3 -m pip install -r ../../huawei_dial/workspace/eval/requirements.txt
cp config.example.env .env
```

Configure the policy service, Qwen user simulator, Elasticsearch index, and
embedding service in `.env`. Variable names match Huawei's
`workspace/eval/config.example.env`; additional timeout and retriever tuning
may be copied there unchanged. Do not place credentials in the example file.

The policy endpoint only needs OpenAI-compatible chat/completions support. It
does not need native function calling because the adapter requests JSON and
emits native tool calls to the Huawei agent loop. Keep
`POLICY_ENABLE_THINKING=false` for a stable structured-action protocol.

## Run

Run a connectivity check first:

```bash
bash run_evaluation.sh --check-only
```

Then run a smoke evaluation. Use a new output directory for every
strategy/model combination:

```bash
bash run_evaluation.sh \
  --limit 20 \
  --workers 4 \
  --output-dir ../../huawei_dial/workspace/eval/outputs/llm-proactive-smoke
```

All ordinary flags are forwarded directly to Huawei's `evaluate.py`, including
`--domains`, `--user-simulator random`, `--resume`, `--aggregate-only`, and
`--skip-judge`. Pass `--huawei-eval-dir /path/to/workspace/eval` only when the
Huawei checkout is elsewhere, or set `HUAWEI_CLARQ_EVAL_DIR`.

Huawei writes `trajectories.jsonl`, `metrics.json`, `report.md`, and
`judge_success.json`. Each non-infrastructure trajectory additionally contains:

```json
"proactive_policy": {
  "name": "llm-proactive-clarq",
  "decisions": [
    {
      "turn": 1,
      "decision": {
        "state": "needs_clarification",
        "missing_information": "device model",
        "basis": "..."
      },
      "action": {
        "name": "clarify_user",
        "arguments": {"question": "..."}
      }
    }
  ]
}
```

`run_config.json` stores the adapter and prompt versions. Resume validation
includes this metadata, so native-policy and LLM-Proactive trajectories cannot
be mixed in one output directory.

## Local Tests

The tests mock the policy, retriever, simulator, and judges, with no network
calls:

```bash
python3 -m unittest discover -s tests -v
```
