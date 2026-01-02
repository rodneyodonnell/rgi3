#!/usr/bin/env bash
# Run integration tests for the full AlphaZero training pipeline
#
# Usage:
#   ./scripts/test_integration.sh              # Run all integration tests
#   ./scripts/test_integration.sh quick        # Run only the quick smoke test
#   ./scripts/test_integration.sh <test_name>  # Run specific test

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running AlphaZero Integration Tests${NC}"
echo ""

if [ "$1" = "quick" ]; then
    echo -e "${GREEN}Running quick smoke test (~5 seconds)...${NC}"
    uv run pytest tests/rgizero/test_integration.py::test_full_training_pipeline_count21 -v -s
elif [ -z "$1" ]; then
    echo -e "${GREEN}Running all integration tests (~10-15 minutes)...${NC}"
    uv run pytest tests/rgizero/test_integration.py -v -s
else
    echo -e "${GREEN}Running test: $1${NC}"
    uv run pytest tests/rgizero/test_integration.py::$1 -v -s
fi

echo ""
echo -e "${GREEN}✓ Integration tests completed${NC}"
