#!/bin/bash
set -e

# Setup trap to kill child processes on exit
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Resolve local Node.js — prefer /home/tbmmai/node-local, fall back to whatever is on PATH
NODE_LOCAL="/home/tbmmai/node-local/bin"
if [ -d "$NODE_LOCAL" ]; then
    export PATH="$NODE_LOCAL:$PATH"
    NPM="$NODE_LOCAL/node $NODE_LOCAL/npm"
else
    NPM="npm"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Starting Backend (FastAPI)..."
export PYTHONPATH="$PROJECT_ROOT"
"$PROJECT_ROOT/.venv/bin/python3" -m uvicorn src.api.server:app \
    --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

WEBAPP="$PROJECT_ROOT/src/ui/web_app"

if [ -d "$WEBAPP" ]; then
    echo "Installing React frontend dependencies..."
    cd "$WEBAPP"
    $NPM install --prefer-offline
    echo "Starting React frontend (Vite)..."
    $NPM run dev &
    FRONTEND_PID=$!
    cd "$PROJECT_ROOT"
    echo "Services started:"
    echo "  - FastAPI backend: http://localhost:8000"
    echo "  - React frontend:  http://localhost:3000"
else
    echo "WARNING: web_app directory not found at $WEBAPP"
    echo "FastAPI backend is serving at: http://localhost:8000"
fi

echo "Press Ctrl+C to stop."

# Wait for background jobs
wait
