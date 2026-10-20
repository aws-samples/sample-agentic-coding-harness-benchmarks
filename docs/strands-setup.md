# Running the Strands Agents harness

`--agent strands` drives a benchmark task with an agent built on the [Strands Agents](https://strandsagents.com) Python SDK. It runs the same `/swe3` task, writes the same six artifacts, and goes through the same judge as the other harnesses. This page covers installing it, how it differs from the CLI harnesses, and a first run.

## How it differs from the other harnesses

Strands is a library, not a command-line agent, so the repo supplies the agent process: [benchmarks/scripts/strands_agent_runner.py](../benchmarks/scripts/strands_agent_runner.py). The headless runner starts that script as a subprocess, just as it starts `pi -p` or `codex exec`. The script builds a Strands `Agent` from four parts:

| Part | What the runner uses |
| --- | --- |
| Model | `BedrockModel` for `provider=bedrock`; `OpenAIModel` for `provider=endpoint` (vLLM or the LiteLLM proxy) |
| Tools | The SDK's built-in `shell` and `file_editor`, and nothing else |
| Skill | The full `.claude/skills/swe3/SKILL.md`, placed ahead of the task prompt with the same wording omp gets. The runner does not use the SDK's `AgentSkills` plugin (see below) |
| Context | `SummarizingConversationManager`, which keeps the task prompt and summarizes older turns. With `--context-window` set it compresses at `auto_compact_fraction` of the window, before the model overflows |

On Amazon Bedrock the runner turns on prompt caching with `CacheConfig(strategy="auto")`. Strands adds cache points only for models that support them, so the setting is safe for every Bedrock model.

The runner writes JSON lines to stdout: a `model_call` event after each model response, a `tool_call` event after each tool call, and one `result` event at the end with the token totals. The harness saves the stream to `strands-stream.jsonl` in the task's artifact folder and reads the totals from the `result` event.

## Install

Strands is an optional dependency group of the `benchmarks/` uv project, so the other harnesses never install it:

```bash
cd benchmarks
uv sync --group strands
uv run --group strands python -c "import importlib.metadata as m; print(m.version('strands-agents'))"   # >= 1.57.1
```

`run-e2e-benchmark.sh --agent strands` installs the group itself when it is missing, and starts the harness with `uv run --group strands`. If you run `run-swe-headless.py` by hand, add `--group strands` to that `uv run` too: a plain `uv sync` removes the group from the venv, and the harness then stops with an error naming the fix.

Nothing else needs installing. There is no binary on `PATH` and no per-user config directory.

## Credentials and routing

- **Amazon Bedrock** (`--provider bedrock`): the runner uses the standard AWS credential chain and the region from `--aws-region`, `aws_region` in the runner config, or `AWS_REGION`. Any Bedrock model id works, including cross-region inference profiles such as `us.anthropic.claude-sonnet-5`.
- **OpenAI-compatible endpoint** (`--provider vllm` or `litellm` in the e2e script): the runner appends `/v1` to the endpoint. The harness passes `api_key` through the `STRANDS_API_KEY` environment variable, so the key never appears in the process list.

## First run

Prove the path on the one-task sanity dataset before a long run:

```bash
cd benchmarks
./scripts/run-e2e-benchmark.sh --provider bedrock --agent strands \
  --model us.anthropic.claude-haiku-4-5-20251001-v1:0 --dataset dataset/hello-world.yaml
```

Artifacts land under `swe-benchmark-data/claude-haiku-4-5/strands/swe3/Hello-World/`. A batch uses the same flag: `run-multi-model-benchmark.sh <model> ... --agent strands`.

From Claude Code, the `/benchmark` skill takes the same choice: ask for `agent=strands`.

## Token and cost accounting

The runner reports the SDK's accumulated usage. The two providers define `inputTokens` differently, and the harness corrects for it:

- Amazon Bedrock reports fresh input tokens, with cache reads and cache writes as separate counts. The harness uses them as they are.
- The OpenAI-compatible client copies the endpoint's `prompt_tokens`, which includes cached tokens, into `inputTokens`, and reports the cached part again as `cacheReadInputTokens`. The harness subtracts the cached part, so the cached prompt is not billed twice.

Strands reports no dollar cost, so the harness falls back to the local price table in [bedrock_pricing.py](../benchmarks/scripts/bedrock_pricing.py), as it does for codex. That table deliberately carries only the GPT models: an Anthropic model reports its cost through the agent, so there is nothing to derive for the agents that pass it through. Strands is not one of those agents -- it reports usage but no cost -- so a Claude model driven by `--agent strands` or `--agent strands-harness` records `total_cost_usd` as **null**, and the token counts are the measurement to use. Price the run yourself from `metrics_that_matter` if you need a figure. `num_turns` is the SDK's event-loop cycle count: one model call and the tool calls it asked for.

## Where the runner departs from Strands as it ships

The runner changes Strands in seven places. Without the first three, Strands could not finish long tasks on a vLLM endpoint. The next two make the Strands score comparable with omp's. The first five came from the `minicpm5-2b` run on vLLM (issue #196); the sixth came from a Claude Haiku 4.5 task that crashed with `RecursionError`; the seventh came from the `minicpm5-2b` run under `--agent strands-harness`.

| Change | Why | Remove it when |
|---|---|---|
| The runner passes `--context-window` minus `--max-tokens` to `OpenAIModel` as `context_window_limit` (issue #197) | Without it, Strands assumes a 200K window and starts summarizing at 180K tokens, past the end of a 128K window. vLLM also counts the requested output tokens against the window, so the prompt only gets what is left | Never. The Bedrock branch already did this |
| An `OpenAIModel` subclass drops `"tools": []` from requests (issue #198) | Strands sends an empty `tools` array on a summary request, and vLLM rejects it with HTTP 400, so summarization always fails | [strands-agents/harness-sdk#4854](https://github.com/strands-agents/harness-sdk/issues/4854) ships a fix |
| An `AfterToolCallEvent` hook cuts any tool result over `MAX_TOOL_RESULT_CHARS` (100,000 characters, about 25K tokens) and tells the model to read a smaller part (issue #199) | `file_editor view` returns whole files up to 1 MB. Two views of 280 KB and 220 KB files overflowed a 128K window in one turn, and the summarizer cannot help while the conversation has 10 or fewer messages | Strands caps tool output itself, or [strands-agents/harness-sdk#3723](https://github.com/strands-agents/harness-sdk/pull/3723) ships and makes the overflow recoverable |
| The runner puts the full `SKILL.md` ahead of the task prompt, worded as omp gets it, and does not load the `AgentSkills` plugin (issue #201) | With the plugin, the model sees only the skill's name and has to call the `skills` tool to read it. `minicpm5-2b` skipped that call on 5 of 12 attempts, and three of those wrote the wrong files or none. Text in the first message also survives summarization, and a tool result does not | Never, while the benchmark compares Strands with omp and kiro, which get the skill the same way |
| The system prompt maps the tool names the skill uses (`Read`, `Edit`, `Write`, `Bash`, `Grep`, `Glob`, `Task`) to the two tools the agent has, `file_editor` and `shell` (issue #202) | `swe3/SKILL.md` was written for Claude Code and names its tools 29 times. vLLM drops a call to a tool that is not in the request, so the call reaches Strands as text and the attempt ends. With the skill in the first message, 4 of the first 6 attempts ended this way | Never, while the skill names Claude Code tools |
| `main()` raises Python's recursion limit to `RECURSION_LIMIT`, 3,500 frames, sized for `MAX_TOOL_TURNS` = 1,000 tool turns (issue #208) | Strands' event loop calls itself after every round of tool calls and adds 3 frames per turn, so the default limit of 1,000 frames ends a task after about 320 turns. A task that runs past 1,000 turns still stops with `RecursionError` | Strands replaces the recursion with a loop |
| A loop guard (`_LoopGuard`) counts every tool call by tool name and exact input. Past `LOOP_WARN_REPEATS` (20) it cancels the repeated call and tells the model to move on; at `LOOP_STOP_REPEATS` (40) it ends the attempt cleanly, so the harness scores the artifacts already written. Each action is a `loop_guard` event in `strands-stream.jsonl` | Under `--agent strands-harness`, `minicpm5-2b` repeated `git diff HEAD --stat` 674 times in one task and a `grep` 542 times in another; the second ran into the recursion limit with no artifacts. Across 42 task attempts, no healthy task repeated an identical call more than 16 times | Never; it only fires on a loop |

The cap changes what a model can do in a turn. A model that reads a 280 KB file sees the first 100,000 characters and a note, where unmodified Strands would have ended the task. Claude Code's `Read` tool stops at about the same size, so the cap brings Strands closer to the other harnesses. Each `tool_call` event in `strands-stream.jsonl` records `truncated_from_chars`, so a reader can count how often the cap fired.

## The Strands harness: `--agent strands-harness`

`--agent strands-harness` runs the same runner script with `--harness`, which builds the agent with the Strands team's [`create_harness()`](https://strandsagents.com/docs/user-guide/harness/) from the `strands-harness` package in place of the hand-built `Agent` above. The package is part of the `strands` dependency group, so `uv sync --group strands` installs both. Results land in their own `<model>/strands-harness/` folder, so they never mix with `--agent strands` runs.

`create_harness()` returns a plain `strands.Agent` with the Strands team's defaults. The runner keeps some of them and turns others off, so a run stays single-agent and isolated like the other harnesses:

| Harness default | In a benchmark run | Why |
| --- | --- | --- |
| `shell`, `read`, `write`, `edit` tools | kept | The coding tools |
| Context manager (`"auto"`), which offloads large tool results to disk and gives the model `retrieve_offloaded_content` | kept | It replaces our 100,000-character cap (#199), which the runner turns off in this mode |
| Prompt caching, skills, the `todos` and `environment` plugins | kept | As shipped. `environment` adds the platform, the date, the working directory and the repository's `AGENTS.md` before each turn |
| `web_fetch`, `web_search` | off | The agent could look up the upstream fix |
| `subagent`, `programmatic_tool_caller` | off | `/swe3` is single-agent |
| `memory`, `session` | off | Nothing may carry over between runs or tasks |

The runner adds three things on top:

- Its own model instance, so the empty-`tools` fix (#198) and the context-window limit (#197) still apply. The harness uses a pre-built model as given.
- The tool map (#202), in the harness `instructions` and again at the top of the first message. In the instructions alone it sits after the harness's own system prompt, and a small model called `bash` on every attempt.
- A `bash` tool: `make_shell(name="bash")`, the same shell the harness builds, under the name small models reach for. With the map in the first message, `minicpm5-2b` still called `bash` once in 44 turns, and an unknown name ends the attempt.

The recursion limit (#208) and the loop guard apply too, since the harness runs the same SDK event loop.

The Strands team also ships the harness as a CLI (`npm install -g @strands-agents/cli`, then `strands -p "<task>"`). The benchmark does not use it: print mode reports no token counts, the CLI loads MCP servers from `~/.claude.json` and other agents' config files on its own, and it is the TypeScript implementation.

## Known limits

- **No turn cap.** Like pi, omp and codex, the Strands agent runs until it stops or the harness's `timeout_seconds` deadline kills it. `max_turns` does not apply.
- **Parallel tool calls run at the same time.** The SDK's default executor runs every tool call in one model response at once. A model that creates a file and reads it back in the same response can race itself. The runner keeps the default so the benchmark measures Strands as it ships.
- **Usage in `model_call` events trails by one call.** The SDK adds a call's usage after the hook that writes the event fires. The `result` event is exact; the running figure only matters when a run is killed before it finishes.

## Related

- [how-a-run-works.md](how-a-run-works.md): the harness flow every agent shares.
- [codex-setup.md](codex-setup.md): the other harness whose cost comes from the price table.
- [benchmarks/docs/harness-reference.md](../benchmarks/docs/harness-reference.md): the dataset format, the artifacts, and the judge.
