# What a single benchmark run does

One task from start to finish: clone the repo, drive the agent, record the metrics, score the artifacts.

## What a single benchmark run does

The flow below is identical across all three paths; only the box the request lands in (Bedrock's Anthropic route, the LiteLLM proxy, or your vLLM server) changes.

```mermaid
sequenceDiagram
    participant H as Harness<br/>(run-swe-headless.py)
    participant G as GitHub repo
    participant CC as Claude Code<br/>(/swe skill)
    participant M as Model<br/>(path 1/2/3)
    participant J as Judge<br/>(codex_judge.py)

    H->>G: clone repo at pinned ref (temp dir)
    H->>CC: claude -p "/swe repo … problem … model …"
    loop agent loop (bounded by max_turns)
        CC->>M: Anthropic Messages request
        M-->>CC: reply (text and/or tool_use)
        CC->>CC: run tools (read repo, write artifacts)
    end
    CC-->>H: 4 artifacts + JSON result (tokens, latency, turns)
    H->>H: write metrics.json beside artifacts
    H->>G: remove temp clone
    J->>J: score the 4 artifacts against the rubric
    J-->>H: eval.json (quality scores) merged into metrics.json
```

The `/swe2` and `/swe3` skills land **six artifacts** -- four design docs plus the implemented change (`patch.diff`, `implementation.md`) -- but they do **not** run tests or open PRs; whether the design and code are any good is the downstream evaluation the judge (or a human) performs on the artifacts. Full mechanics are in the [harness reference](../benchmarks/docs/harness-reference.md).

> **"SWE" here means software engineering in general -- not [SWE-bench](https://www.swebench.com/), the specific benchmark dataset.** The `/swe` skill lets you run any model against any task in any repo of your choosing. It is a *harness*, not a fixed benchmark set: compare results across models on the same task, or a single model across tasks of varying difficulty.


---

[< Back to the README](../README.md)
