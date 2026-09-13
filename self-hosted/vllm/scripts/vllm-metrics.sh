#!/usr/bin/env bash
set -euo pipefail

# Run the continuous vLLM Prometheus collector independently of benchmarks.
#
# Usage:
#   ./vllm-metrics.sh start
#   ./vllm-metrics.sh status
#   ./vllm-metrics.sh stop
#   ./vllm-metrics.sh foreground
#
# Environment variables:
#   BASE_URL          vLLM OpenAI base URL      (default: http://127.0.0.1:8000/v1)
#   METRICS_URL       explicit /metrics URL     (default: derived from BASE_URL)
#   METRICS_INTERVAL  seconds between scrapes   (default: 1)
#   METRICS_DB        DuckDB output path         (default: benchmark-output/vllm-metrics.duckdb)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$VLLM_DIR/benchmark-output"
LOG_DIR="$VLLM_DIR/logs"
PID_FILE="${METRICS_PID_FILE:-/tmp/vllm-metrics.pid}"
LOG_FILE="${METRICS_LOG_FILE:-$LOG_DIR/vllm-metrics.log}"
DATABASE="${METRICS_DB:-$OUTPUT_DIR/vllm-metrics.duckdb}"
INTERVAL="${METRICS_INTERVAL:-1}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

info() { printf '[info]  %s\n' "$1"; }
ok() { printf '[ok]    %s\n' "$1"; }
fail() { printf '[fail]  %s\n' "$1" >&2; exit 1; }

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE")"
  kill -0 "$pid" 2>/dev/null
}

start_collector() {
  [[ -n "$UV_BIN" ]] || fail "uv is not installed or not on PATH"
  if is_running; then
    ok "Metrics collector is already running (PID $(cat "$PID_FILE"))."
    return
  fi

  mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
  rm -f "$PID_FILE"
  info "Starting metrics collector: $BASE_URL -> $DATABASE"
  (
    cd "$VLLM_DIR"
    nohup "$UV_BIN" run python -m clients.collect_metrics \
      --base-url "$BASE_URL" \
      --database "$DATABASE" \
      --interval "$INTERVAL" \
      >"$LOG_FILE" 2>&1 &
    echo "$!" > "$PID_FILE"
  )

  sleep 1
  is_running || fail "Collector exited during startup. Check $LOG_FILE"
  ok "Metrics collector running (PID $(cat "$PID_FILE"), interval ${INTERVAL}s)."
  info "Database: $DATABASE"
  info "Log:      $LOG_FILE"
}

stop_collector() {
  if ! is_running; then
    rm -f "$PID_FILE"
    info "Metrics collector is not running."
    return
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  kill "$pid"
  for _ in $(seq 1 50); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid"
  fi
  rm -f "$PID_FILE"
  ok "Metrics collector stopped."
}

show_status() {
  if is_running; then
    ok "Metrics collector is running (PID $(cat "$PID_FILE"))."
    info "Database: $DATABASE"
    info "Log:      $LOG_FILE"
  else
    fail "Metrics collector is not running."
  fi
}

run_foreground() {
  [[ -n "$UV_BIN" ]] || fail "uv is not installed or not on PATH"
  mkdir -p "$OUTPUT_DIR"
  cd "$VLLM_DIR"
  exec "$UV_BIN" run python -m clients.collect_metrics \
    --base-url "$BASE_URL" \
    --database "$DATABASE" \
    --interval "$INTERVAL"
}

case "${1:-start}" in
  start) start_collector ;;
  stop) stop_collector ;;
  status) show_status ;;
  foreground) run_foreground ;;
  -h|--help)
    sed -n '3,17p' "$0" | sed 's/^# \{0,1\}//'
    ;;
  *) fail "Unknown command: $1 (expected start, status, stop, or foreground)" ;;
esac
