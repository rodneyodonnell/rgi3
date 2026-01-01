#!/bin/bash
# Compile frontend first
./scripts/compile_frontend.sh

# Start Server
uv run uvicorn web_app.app.main:app --host 0.0.0.0 --port 8000 --reload
