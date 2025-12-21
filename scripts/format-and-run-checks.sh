#!/bin/bash
set -e

# Parse arguments
CHECK_MODE=0
for arg in "$@"; do
    if [ "$arg" == "--check" ]; then
        CHECK_MODE=1
        break
    fi
done

if [ $CHECK_MODE -eq 1 ]; then
    echo "Running nbstripout check..."
    # nbstripout doesn't have a simple --check flag that returns exit code 1
    # so we check if dry-run produces any output
    OUTPUT=$(uv run nbstripout --dry-run $(find . -iname '*.ipynb'))
    if [ -n "$OUTPUT" ]; then
        echo "The following notebooks are not stripped:"
        echo "$OUTPUT"
        echo "Please run './scripts/format-and-run-checks.sh' to fix."
        exit 1
    fi
else
    echo "Running nbstripout..."
    uv run nbstripout $(find . -iname '*.ipynb')
fi

if [ $CHECK_MODE -eq 1 ]; then
    echo "Running ruff format check..."
    uv run ruff format --check .
else
    echo "Running ruff-format..."
    uv run ruff format .
fi

echo "Running ruff check..."
uv run ruff check .

echo "Running type checker..."
uv run ty check .
