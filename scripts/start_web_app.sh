#!/bin/bash
PORT=8000

# Check if port is in use
PID_LIST=$(lsof -t -i :$PORT)
if [ -n "$PID_LIST" ]; then
    echo "❌ Error: Port $PORT is already in use."
    # Flatten newlines to spaces for display
    PIDS_ONeline=$(echo $PID_LIST | xargs)
    echo "Process IDs: $PIDS_ONeline"
    echo ""
    echo "To kill these processes, run:"
    echo "  kill $PIDS_ONeline"
    exit 1
fi

# Compile frontend first
./scripts/compile_frontend.sh

# Start Server
uv run uvicorn web_app.app.main:app --host 0.0.0.0 --port $PORT --reload
