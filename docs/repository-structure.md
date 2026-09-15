# Repository structure

What lives where in this repository.

## Repository structure

```text
sample-agentic-coding-harness-benchmarks/
├── README.md                  ← Start here (concepts, the two steps, the three hosting paths)
├── LICENSE                    MIT-0
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── SECURITY.md
├── SUPPORT.md
├── THIRD_PARTY                Third-party dependency attributions
├── .github/                   Issue and pull-request templates
├── .claude/                   ← Claude Code skills shipped with the repo
│   └── skills/
│       ├── setup-machine/     /setup-machine — inspect a fresh box, install every dependency (start here)
│       ├── benchmark/         /benchmark — run one end-to-end benchmark (service + harness + judge)
│       ├── swe/, swe2/, swe3/ /swe* — drive a model through a SWE task on any repo (swe3 is the default)
│       ├── swe-router/      /swe-router — recommend the right model for a task, from these measurements
│       ├── throughput/        /throughput — sweep a served model's throughput
│       ├── security-check/    /security-check — Cipher security review + fix before any commit
│       └── vllm-setup/        /vllm-setup — stand up the EC2 vLLM server (Path 3)
├── docs/                      ← Concepts, setup guides, and methodology
├── vend/                      ← Vendored, installable artifacts
│   └── swe-router/            /swe-router skill: route.py, models.json, install.sh
├── benchmarks/                ← The benchmark harness
│   ├── README.md              Harness landing page
│   ├── docs/                  Shared harness reference + one guide per hosting path
│   ├── config/                runner.example.yaml, litellm-mantle.yaml (Path 2 proxy)
│   ├── dataset/               Benchmark dataset YAML files
│   ├── scripts/               Run harness, dataset/config loaders, judges, proxy launcher
│   ├── tests/                 Unit tests
│   └── swe-benchmark-data/    Where your runs land. Gitignored — no run output is committed.
└── self-hosted/               ← Path 3: EC2 self-hosted serving (vLLM)
    └── vllm/
        ├── README.md          Full EC2 + vLLM setup guide
        ├── models/            Per-model serving guidelines (one .md per model)
        ├── scripts/           vllm-install.sh, vllm-serve.sh, tunnel.sh, …
        ├── clients/           Inference + metrics-collection Python clients
        ├── tests/             unittest suite for the clients
        └── config/            claude-code.json, opencode.json
```


---

[< Back to the README](../README.md)
