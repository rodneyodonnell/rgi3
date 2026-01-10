#!/usr/bin/env bash
# Run rgizero and web_app tests
#
# Usage:
#   ./scripts/test_rgizero.sh               # Run fast unit tests only
#   ./scripts/test_rgizero.sh --with-integration  # Include integration tests

set -e

if [ "$1" = "--with-integration" ]; then
    echo "Running all tests (including integration)..."
    uv run pytest tests/rgizero tests/web_app
else
    echo "Running fast tests (excluding integration)..."
    echo "Use --with-integration to include integration tests."
    uv run pytest tests/rgizero tests/web_app -m "not integration"
fi
