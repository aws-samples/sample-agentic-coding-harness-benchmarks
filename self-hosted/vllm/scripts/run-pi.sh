#!/bin/bash
# run-pi.sh - Start pi against whatever model the local vLLM server is serving.
#
# vLLM exposes an OpenAI-compatible API; pi reaches it via the "vllm" custom
# provider declared in ~/.pi/agent/models.json. This script discovers the served
# model id from /v1/models so you never hand-edit JSON per model. It also syncs
# the provider's baseUrl and its single anchor model id in models.json to match
# the running server, so even a bare `pi` (no wrapper) resolves to a real model
# instead of a stale placeholder that 404s.
#
# Usage:
#   ./run-pi.sh                       # auto-detect model, start interactive pi
#   ./run-pi.sh -p "fix the bug"      # extra args are forwarded to pi
#   MODEL=glm-5.2 ./run-pi.sh         # override auto-detection
#   BASE_URL=http://host:8000/v1 ./run-pi.sh
#
# Env:
#   BASE_URL   vLLM OpenAI-compatible endpoint (default: http://127.0.0.1:8000/v1)
#   MODEL      served model id to use (default: first id reported by the server)
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-}"
MODELS_JSON="${PI_MODELS_JSON:-$HOME/.pi/agent/models.json}"

# Discover the served model id when not overridden.
if [ -z "$MODEL" ]; then
  models_json="$(curl -fsS --max-time 5 "$BASE_URL/models" 2>/dev/null || true)"
  if [ -z "$models_json" ]; then
    echo "run-pi.sh: no response from vLLM at $BASE_URL/models" >&2
    echo "           is the server up? try: curl $BASE_URL/models" >&2
    exit 1
  fi
  MODEL="$(printf '%s' "$models_json" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin).get("data",[]); print(d[0]["id"] if d else "")')"
  if [ -z "$MODEL" ]; then
    echo "run-pi.sh: server returned no models at $BASE_URL/models" >&2
    exit 1
  fi
fi

# Sync models.json so the vllm provider's baseUrl and anchor model id match the
# running server. Keeps a single-model provider block; a bare `pi` then resolves
# to the real served id instead of a stale placeholder.
MODEL="$MODEL" BASE_URL="$BASE_URL" MODELS_JSON="$MODELS_JSON" python3 <<'PY'
import json, os, pathlib

path = pathlib.Path(os.environ["MODELS_JSON"])
model = os.environ["MODEL"]
base_url = os.environ["BASE_URL"]

data = {}
if path.exists():
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        data = {}

providers = data.setdefault("providers", {})
vllm = providers.setdefault("vllm", {})
vllm["baseUrl"] = base_url
vllm.setdefault("api", "openai-completions")
vllm.setdefault("apiKey", "local")
vllm.setdefault("compat", {"supportsDeveloperRole": False, "supportsReasoningEffort": False})
vllm["models"] = [{
    "id": model,
    "name": f"vLLM: {model}",
    "reasoning": False,
    "input": ["text"],
    "contextWindow": 200000,
    "maxTokens": 20000,
    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
}]

path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, indent=2) + "\n")
PY

echo "run-pi.sh: using model '$MODEL' at $BASE_URL" >&2
exec pi --provider vllm --model "$MODEL" "$@"
