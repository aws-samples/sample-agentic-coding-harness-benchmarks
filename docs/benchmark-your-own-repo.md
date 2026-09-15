# Benchmark your own repositories

The harness works against any GitHub repository. Name your own repos in a dataset YAML and the models, judge and cost math are the ones behind every published result here.

## Datasets

A dataset is a single YAML file: a metadata header plus a list of tasks, each pointing at a GitHub repo and a problem. Two datasets ship in [benchmarks/dataset/](../benchmarks/dataset/):

- [hello-world.yaml](../benchmarks/dataset/hello-world.yaml) -- a trivial sanity dataset (the [octocat/Hello-World](https://github.com/octocat/Hello-World) repo) for kicking the tires of a new model or endpoint.
- [mcp-gateway-registry.yaml](../benchmarks/dataset/mcp-gateway-registry.yaml) -- the reference dataset, whose tasks are drawn from real upstream issues in [agentic-community/mcp-gateway-registry](https://github.com/agentic-community/mcp-gateway-registry).

**Nothing in the harness is specific to a particular repository.** Adding your own benchmark dataset is just writing another YAML file in the same format -- point tasks at any public repo and pinned ref. The dataset format is documented in the [harness reference](../benchmarks/docs/harness-reference.md#the-dataset).

### What do I do with this?

The point of this repo is to help you **pick the right coding agent and model for your tasks** -- the pairing that lands the quality you need at the cost and latency you can live with, instead of defaulting to the most expensive option. There are two ways to get there:

1. **Start from the reference frontier that ships with `swe-router`.** [`vend/swe-router/models.json`](../vend/swe-router/models.json) carries a measured per-tier score and cost per task for each model, so the skill can make a recommendation out of the box. Treat it as a starting point: it was measured on one Python/FastAPI service, and its own `single_repository_warning` says applying it to a very different codebase is extrapolation.
2. **Build your own frontier on your own code.** When you want numbers on **work that looks like yours** rather than a reference repo, use the benchmarking harness here: write a dataset YAML pointing at your repositories and run it. The models, harnesses, judge, and cost math are the same ones that produced the reference numbers. This is the rest of this section.

**Then put it in front of developers.** [`swe-router`](../vend/swe-router/) reads whichever frontier you point it at -- ours or the one you just built -- and names the cheapest model clearing the bar for each task. Five files, no dependencies, works in any assistant that reads a skill. Point it at your own `models.json` and the recommendations are grounded in your code rather than our example repo.

### Benchmark your own code repositories

This is option 2 above -- building your own frontier on your own code. It is a few steps:

1. **Create a dataset file** under [benchmarks/dataset/](../benchmarks/dataset/), for example `my-team.yaml`. Copy [mcp-gateway-registry.yaml](../benchmarks/dataset/mcp-gateway-registry.yaml) as a template. Minimal shape:

   ```yaml
   schema_version: "1.0"
   name: my-team
   title: My team's benchmark
   description: Real tasks from our own repositories.
   default_ref: main                      # pin a tag/commit per task for reproducibility
   metrics: [input_tokens, output_tokens, num_turns]
   complexity_levels: [low, medium, high]
   tasks:
     - id: add-rate-limiting-to-gateway
       repo: https://github.com/your-org/your-repo
       ref: v2.3.0                         # pin so every run clones the same code
       complexity: medium
       tags: [python, api, feature]
       problem_statement: |
         Describe the task in enough detail for an agent to act on it without
         you present -- what to change, constraints, and what "done" means.
   ```

Each task points at a repo + pinned ref + a problem statement (from a real ticket or issue). Full field reference: [harness reference -> The dataset](../benchmarks/docs/harness-reference.md#the-dataset). Any repo the runner can `git clone` works (public, or private with credentials available to your shell).

2. **Run it** against whichever model/harness/path you want -- same commands as the example, just swap the dataset:

   ```
   /benchmark provider=bedrock model=claude-opus-5 dataset=dataset/my-team.yaml
   ```

or headless: `benchmarks/scripts/run-e2e-benchmark.sh --provider bedrock --model ... --dataset dataset/my-team.yaml`. Pick the harness with `--agent claude|pi|omp|kiro` and the skill with `--skill swe2|swe3` (`--agent kiro` drives Kiro's managed models and sets `--provider kiro` for you).

3. **Read your results.** Artifacts and scores land under `benchmarks/swe-benchmark-data/<model>/<harness>/<skill>/<your-dataset-repo>/<task>/`, and the same generators build your own cost/quality frontier (`gen_swe_comparison.py`, `plot_cost_quality.py`). Your runs are gitignored, so a customer's private code never lands in version control.

> **Tips for good tasks:** pin a `ref` so reruns are comparable; write the `problem_statement` like a well-scoped ticket; use `tags` to slice results by language/domain/change-type; and add optional `ground_truth` (reviewer-only, never shown to the agent) if you want the judge to check against a known-good approach.


---

[< Back to the README](../README.md)
