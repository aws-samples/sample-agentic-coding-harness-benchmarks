#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# tunnel.sh — Manage SSH tunnel to EC2 GPU instance running vLLM
#
# Usage:
#   ./tunnel.sh start    # Open tunnel (localhost:$LOCAL_MODEL_PORT → ec2:$LOCAL_MODEL_PORT)
#   ./tunnel.sh stop     # Close tunnel
#   ./tunnel.sh status   # Check if tunnel is active
#
# Environment variables:
#   G6E_IP             Required for start — public IP of the EC2 GPU instance
#   G6E_KEY            Path to SSH key (default: ~/.ssh/id_rsa)
#   LOCAL_MODEL_PORT   Local port to forward (default: 8000, vLLM's port)
# ---------------------------------------------------------------------------

G6E_IP="${G6E_IP:-}"
KEY="${G6E_KEY:-$HOME/.ssh/id_rsa}"
LOCAL_PORT="${LOCAL_MODEL_PORT:-8000}"
REMOTE_PORT="${LOCAL_MODEL_PORT:-8000}"

if [[ "${1:-status}" == "start" ]] && [[ -z "$G6E_IP" ]]; then
    echo "Error: G6E_IP not set. Export it first:"
    echo "  export G6E_IP=<your-ec2-instance-public-ip>"
    echo "  export G6E_KEY=<path-to-your-key.pem>  # optional, defaults to ~/.ssh/id_rsa"
    exit 1
fi

case "${1:-status}" in
    start)
        # Validate SSH key exists and has correct permissions
        if [[ ! -f "$KEY" ]]; then
            echo "Error: SSH key not found: $KEY"
            echo "  Set G6E_KEY to the path of your .pem file."
            exit 1
        fi

        KEY_PERMS=$(stat -f "%Lp" "$KEY" 2>/dev/null || stat -c "%a" "$KEY" 2>/dev/null || echo "")
        if [[ -n "$KEY_PERMS" ]] && [[ "$KEY_PERMS" != "400" ]] && [[ "$KEY_PERMS" != "600" ]]; then
            echo "Warning: SSH key permissions are $KEY_PERMS (should be 400 or 600)."
            echo "  Fix with: chmod 600 \"$KEY\""
        fi

        if lsof -ti :"$LOCAL_PORT" >/dev/null 2>&1; then
            echo "Port $LOCAL_PORT already in use. Run './tunnel.sh stop' first."
            exit 1
        fi
        echo "Opening SSH tunnel: localhost:$LOCAL_PORT → $G6E_IP:$REMOTE_PORT"
        ssh -N -f \
            -o StrictHostKeyChecking=accept-new \
            -L "$LOCAL_PORT":localhost:"$REMOTE_PORT" \
            -i "$KEY" ubuntu@"$G6E_IP"
        sleep 2
        if curl -sf "http://localhost:${LOCAL_PORT}/v1/models" >/dev/null 2>&1; then
            echo "Tunnel active. Model reachable at http://localhost:$LOCAL_PORT"
        else
            echo "Tunnel opened but model not responding. Check the EC2 instance (is vLLM running?)."
        fi
        ;;
    stop)
        PIDS=$(lsof -ti :"$LOCAL_PORT" 2>/dev/null || echo "")
        if [[ -n "$PIDS" ]]; then
            echo "Killing tunnel (PIDs: $PIDS)"
            echo "$PIDS" | xargs kill 2>/dev/null || true
            echo "Tunnel closed."
        else
            echo "No tunnel running on port $LOCAL_PORT."
        fi
        ;;
    status)
        if curl -sf "http://localhost:${LOCAL_PORT}/v1/models" >/dev/null 2>&1; then
            echo "Tunnel active. Model responding on localhost:$LOCAL_PORT"
            curl -s "http://localhost:${LOCAL_PORT}/v1/models" | python3 -c "
import json, sys
data = json.load(sys.stdin)
models = data.get('data', [])
for m in models:
    print(f\"  Model: {m.get('id', 'unknown')}\")
"
        else
            echo "No tunnel or model not responding on localhost:$LOCAL_PORT"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        exit 1
        ;;
esac
