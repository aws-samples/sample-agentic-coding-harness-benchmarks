#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# bedrock-mantle-proxy.sh -- start/stop a LiteLLM proxy over Amazon Bedrock's
# OpenAI-compatible `bedrock-mantle` endpoint, so the SWE harness can benchmark
# non-Anthropic Bedrock models (Kimi, Qwen, DeepSeek, ...) with WORKING tool
# calls.
#
# Why this and not `--provider bedrock`? Claude Code speaks the Anthropic
# Messages API. Sent to Bedrock's Converse path, non-Anthropic models return
# their native tool-call tokens (e.g. Kimi's `<|tool_calls_section|>`) as plain
# text, so Claude Code never sees a structured tool_use block and agentic runs
# stall at one turn with 0 artifacts. The `bedrock-mantle` endpoint is
# OpenAI-compatible and parses those into real tool calls; this proxy bridges
# Anthropic /v1/messages -> OpenAI Chat Completions -> bedrock-mantle.
#
# Auth: a 12h bearer token minted from your ambient AWS credentials via
# aws-bedrock-token-generator, injected as MANTLE_API_KEY at proxy startup.
# Clients send a throwaway key; the proxy holds the real one.
#
# Anthropic (Claude) models do NOT need this -- run them with `--provider
# bedrock` directly.
#
# Usage:
#   ./scripts/bedrock-mantle-proxy.sh            # install deps, mint token, start on :4000
#   ./scripts/bedrock-mantle-proxy.sh --port 8080
#   ./scripts/bedrock-mantle-proxy.sh --stop
#   ./scripts/bedrock-mantle-proxy.sh --status
#   ./scripts/bedrock-mantle-proxy.sh --refresh  # remint token; restart to apply
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$BENCHMARKS_DIR/config/litellm-mantle.yaml"
DEFAULT_HOST="127.0.0.1"
DEFAULT_PORT=4000
PID_FILE="$BENCHMARKS_DIR/.litellm.pid"
TOKEN_FILE="$BENCHMARKS_DIR/.mantle-token"
LOG_FILE="$BENCHMARKS_DIR/.litellm.log"
MANTLE_REGION="${AWS_REGION:-us-east-1}"

usage() {
    echo "Usage: $0 [--host HOST] [--port PORT] [--stop] [--status] [--refresh]"
    echo ""
    echo "Options:"
    echo "  --host HOST   Address to bind on (default: $DEFAULT_HOST). Use a"
    echo "                non-loopback address only if clients on other hosts"
    echo "                must reach the proxy; it does NOT authenticate clients."
    echo "  --port PORT   Port to run the proxy on (default: $DEFAULT_PORT)"
    echo "  --stop        Stop the running proxy"
    echo "  --status      Report proxy and token status"
    echo "  --refresh     Remint the Bedrock bearer token (restart to apply)"
    exit 1
}

HOST=$DEFAULT_HOST
PORT=$DEFAULT_PORT
ACTION="start"

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)    HOST="$2"; shift 2 ;;
        --port)    PORT="$2"; shift 2 ;;
        --stop)    ACTION="stop"; shift ;;
        --status)  ACTION="status"; shift ;;
        --refresh) ACTION="refresh"; shift ;;
        -h|--help) usage ;;
        *) echo "[error] Unknown option: $1"; usage ;;
    esac
done

stop_proxy() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            rm -f "$PID_FILE"
            echo "[stopped] LiteLLM proxy (PID $pid)"
        else
            rm -f "$PID_FILE"
            echo "[info] Proxy was not running (stale PID file cleaned)"
        fi
    else
        echo "[info] No proxy running"
    fi
}

check_status() {
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "[running] LiteLLM proxy PID $(cat "$PID_FILE")"
        curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 \
            && echo " - health: OK" || echo " - health: unreachable"
    else
        echo "[stopped] No proxy running"
    fi
    if [[ -f "$TOKEN_FILE" ]]; then
        local age_sec age_hr
        age_sec=$(( $(date +%s) - $(stat -c%Y "$TOKEN_FILE" 2>/dev/null || stat -f%m "$TOKEN_FILE") ))
        age_hr=$(( age_sec / 3600 ))
        echo "[token] Age: ${age_hr}h (valid 12h; refresh with --refresh)"
    else
        echo "[token] No token file"
    fi
}

generate_token() {
    echo "[token] Minting Bedrock bearer token for $MANTLE_REGION..."
    local token
    token=$(AWS_REGION="$MANTLE_REGION" uv run --with aws-bedrock-token-generator python -c "
from aws_bedrock_token_generator import provide_token
print(provide_token(region='${MANTLE_REGION}'))
")
    if [[ -z "$token" ]]; then
        echo "[error] Failed to mint Bedrock token. Check AWS credentials."
        exit 1
    fi
    export MANTLE_API_KEY="$token"
    ( umask 077; echo "$token" > "$TOKEN_FILE" )
    echo "[token] Bearer token minted (valid 12h)"
}

refresh_token() {
    generate_token
    echo "[done] Token refreshed at $TOKEN_FILE."
    echo "[note] A running proxy will NOT pick up the new token automatically;"
    echo "       it is injected via MANTLE_API_KEY at startup. Restart to apply:"
    echo "         $0 --stop && $0"
}

start_proxy() {
    # Verify AWS credentials resolve before doing anything expensive.
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        echo "[error] AWS credentials not configured for $MANTLE_REGION."
        echo "        Run 'aws configure', use SSO, or attach an IAM role with Bedrock access."
        exit 1
    fi

    generate_token
    stop_proxy 2>/dev/null || true

    echo "[start] LiteLLM proxy on ${HOST}:${PORT}"
    echo "[config] $CONFIG_FILE"
    echo "[backend] Amazon Bedrock (bedrock-mantle.${MANTLE_REGION}.api.aws)"
    if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" ]]; then
        echo "[warn] Binding to $HOST -- reachable from any host that can reach"
        echo "[warn] $HOST:$PORT. The proxy does NOT authenticate clients; rely"
        echo "[warn] on your security group / firewall."
    fi

    export MANTLE_API_KEY
    export LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES=true
    setsid uv run --with 'litellm[proxy]>=1.72,<1.84' --with 'fastapi<0.116' litellm \
        --config "$CONFIG_FILE" --host "$HOST" --port "$PORT" \
        > "$LOG_FILE" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"

    echo -n "[wait] Proxy starting"
    local i
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            echo ""
            echo "[ready] Proxy on http://${HOST}:${PORT} (PID $(cat "$PID_FILE"))"
            echo "[log] $LOG_FILE"
            echo ""
            echo "Run the harness against it:"
            echo "  uv run scripts/run-swe-headless.py --config config/runner.yaml \\"
            echo "    --provider endpoint --endpoint http://${HOST}:${PORT} \\"
            echo "    --model moonshotai.kimi-k2-thinking --dataset dataset/hello-world.yaml --stream"
            return 0
        fi
        echo -n "."
        sleep 2
    done

    echo ""
    echo "[error] Proxy did not become healthy in time. Check the log:"
    echo "        tail -f $LOG_FILE"
    exit 1
}

case $ACTION in
    start)   start_proxy ;;
    stop)    stop_proxy ;;
    status)  check_status ;;
    refresh) refresh_token ;;
esac
