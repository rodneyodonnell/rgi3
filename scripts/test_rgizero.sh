#!/usr/bin/env bash
# Run rgizero tests, excluding slow integration tests by default
#
# Usage:
#   ./scripts/test_rgizero.sh               # Run fast unit tests only
#   ./scripts/test_rgizero.sh --with-integration  # Include integration tests

set -e

if [ "$1" = "--with-integration" ]; then
    echo "Running all rgizero tests (including integration tests)..."
    ./scripts/run_tests.sh tests/rgizero
else
    echo "Running fast rgizero tests (excluding integration tests)..."
    echo "Use --with-integration to include integration tests."
    ./scripts/run_tests.sh tests/rgizero -m "not integration"
fi
