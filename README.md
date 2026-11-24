# thinking-8b-qa-repos

**A bounded-memory reasoning harness for repository question answering with small local language models.**

**No model downloads, model loading in Python, or GPU requirement for the harness.** Connect an already available model through LM Studio or Ollama. The 8B name describes the design target, not a model-size requirement.

## Requirements

Requires Python 3.10+ and SQLite with FTS5; verified with Python 3.11.15. No third-party Python runtime dependencies.


## Quick start

Load a model you already have in LM Studio and start its local server. Set its context window to **8,192 tokens in the server**.


Replace `MODEL_ID`, the repository path and the question:

```bash
cd thinking-8b-qa-repos
python3.11 -m venv .venv
source .venv/bin/activate
python -m thinking_8b_qa_repos index /path/to/repository --chunk-bytes 1000
python -m thinking_8b_qa_repos ask /path/to/repository \
  'Where is the session checked, and how are expired tokens handled?' \
  --model MODEL_ID --session ./session-name \
  --context-window 8192 --prompt-bytes 6000 --output-tokens 1024 \
  --memory-bytes 2048 --evidence-bytes 1800 --max-steps 24
```

Optionally install the CLI entry point:

```bash
python -m pip install -e .
thinking-8b-qa-repos --help
```


## Ollama

For Ollama, add `--provider ollama --base-url http://127.0.0.1:11434`. Use `--no-structured` if the server does not support JSON Schema. Local validation still applies; a model unable to follow the output format can fail the workflow.

Set `THINKING_8B_QA_REPOS_API_KEY` for authenticated servers. Endpoints are local by default; sending repository content to a remote endpoint requires `--allow-remote`.

## Reasoning workflow

1. **Index without an LLM.**
2. **Plan an investigation.**
3. **Read bounded evidence.**
4. **Record grounded findings.**
5. **Control memory access.**
6. **Synthesize every finding.**
7. **Audit in batches.**

## Recover a failed synthesis

**To recover the session that failed:** update the project code, then repeat the exact original `ask` command, keeping its question, repository, budgets, memory policy and `--session`, and add:

```text
--resume --synthesize-only
```