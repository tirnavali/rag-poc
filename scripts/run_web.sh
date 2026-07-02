#!/usr/bin/env bash
# Dev launcher for the web UI (FastAPI backend + Vite frontend).
#
# Robust teardown: uvicorn --reload and `npm run dev` both spawn grandchild
# processes (a multiprocessing worker / a `node vite` under `sh`) that routinely
# survive Ctrl+C and keep the port bound. This script (a) clears stale listeners
# on the target ports before starting, and (b) on exit guarantees both ports are
# actually freed — SIGTERM first, then SIGKILL for stragglers.
#
# NOTE: no `set -e` on purpose — a failed startup step must still run cleanup.
set -uo pipefail

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
RELOAD="${RELOAD:-1}"   # set RELOAD=0 to disable uvicorn auto-reload (steadier for demos)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEBAPP="$PROJECT_ROOT/src/ui/web_app"

# Resolve local Node.js — prefer /home/tbmmai/node-local, fall back to PATH.
NODE_LOCAL="/home/tbmmai/node-local/bin"
[ -d "$NODE_LOCAL" ] && export PATH="$NODE_LOCAL:$PATH"

# List PIDs listening on a TCP port (lsof, with fuser as fallback).
listeners_on() {
    local port="$1" pids
    pids="$(lsof -ti "tcp:${port}" -sTCP:LISTEN 2>/dev/null || true)"
    if [ -z "$pids" ] && command -v fuser >/dev/null 2>&1; then
        pids="$(fuser -n tcp "$port" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' || true)"
    fi
    echo "$pids"
}

# Ensure nothing is left listening on a port: SIGTERM, wait, then SIGKILL.
free_port() {
    local port="$1" pids
    pids="$(listeners_on "$port")"
    [ -z "$pids" ] && return 0
    echo "  port ${port} busy → stopping $(echo "$pids" | tr '\n' ' ')"
    kill $pids 2>/dev/null || true
    for _ in 1 2 3 4 5 6; do
        sleep 0.5
        pids="$(listeners_on "$port")"
        [ -z "$pids" ] && return 0
    done
    echo "  port ${port} still held → SIGKILL $(echo "$pids" | tr '\n' ' ')"
    kill -9 $pids 2>/dev/null || true
    sleep 0.3
}

CLEANED=0
cleanup() {
    [ "$CLEANED" = 1 ] && return
    CLEANED=1
    trap - SIGINT SIGTERM EXIT   # disarm to avoid re-entry
    echo
    echo "Stopping services..."
    # Graceful: signal the whole process group (script + direct children).
    kill -- -$$ 2>/dev/null || true
    sleep 0.5
    # Guarantee: reclaim the ports even if a reload/vite grandchild was orphaned.
    free_port "$BACKEND_PORT"
    free_port "$FRONTEND_PORT"
    echo "Done. Ports ${BACKEND_PORT} and ${FRONTEND_PORT} are free."
}
trap cleanup SIGINT SIGTERM EXIT

echo "Pre-flight: clearing any stale listeners on :${BACKEND_PORT} and :${FRONTEND_PORT} ..."
free_port "$BACKEND_PORT"
free_port "$FRONTEND_PORT"

echo "Starting Backend (FastAPI) on :${BACKEND_PORT} ..."
export PYTHONPATH="$PROJECT_ROOT"
UVICORN_ARGS=(-m uvicorn src.api.server:app --host 127.0.0.1 --port "$BACKEND_PORT")
[ "$RELOAD" = 1 ] && UVICORN_ARGS+=(--reload)
"$PROJECT_ROOT/.venv/bin/python3" "${UVICORN_ARGS[@]}" &
BACKEND_PID=$!

if [ -d "$WEBAPP" ]; then
    echo "Installing React frontend dependencies..."
    ( cd "$WEBAPP" && npm install --prefer-offline )
    echo "Starting React frontend (Vite) on :${FRONTEND_PORT} ..."
    ( cd "$WEBAPP" && exec npm run dev ) &
    FRONTEND_PID=$!
    echo
    echo "Services started:"
    echo "  - FastAPI backend: http://localhost:${BACKEND_PORT}"
    echo "  - React frontend:  http://localhost:${FRONTEND_PORT}"
else
    echo "WARNING: web_app directory not found at $WEBAPP"
    echo "FastAPI backend is serving at: http://localhost:${BACKEND_PORT}"
fi

echo
echo "Press Ctrl+C to stop."
wait
