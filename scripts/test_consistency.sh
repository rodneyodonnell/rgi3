#!/usr/bin/env bash
# Run integration tests multiple times in parallel to verify consistency
#
# Usage:
#   ./scripts/test_consistency.sh                    # Run 10 parallel test runs
#   ./scripts/test_consistency.sh 5                  # Run 5 parallel test runs
#   ./scripts/test_consistency.sh 10 count21         # Test specific game

set -e

NUM_RUNS=${1:-10}
GAME=${2:-"count21"}

echo "Running $NUM_RUNS parallel integration test runs for $GAME..."
echo "This helps verify the tests pass consistently and not just due to luck."
echo

# Create temp directory for logs
LOGDIR=$(mktemp -d)/consistency-tests
mkdir -p "$LOGDIR"

echo "Logs will be saved to: $LOGDIR"
echo

# Launch parallel test runs
for i in $(seq 1 $NUM_RUNS); do
    (
        echo "Starting run $i..."
        uv run pytest tests/rgizero/test_integration.py::test_elo_progression_across_generations \
            -v -s \
            > "$LOGDIR/run-$i.log" 2>&1
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✓ Run $i: PASSED"
        else
            echo "✗ Run $i: FAILED (exit code $EXIT_CODE)"
        fi
        exit $EXIT_CODE
    ) &
done

# Wait for all background jobs
wait

# Count successes and failures
PASSED=0
FAILED=0

for i in $(seq 1 $NUM_RUNS); do
    if grep -q "1 passed" "$LOGDIR/run-$i.log" 2>/dev/null; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

# Print summary
echo
echo "=================================="
echo "CONSISTENCY TEST RESULTS"
echo "=================================="
echo "Total runs: $NUM_RUNS"
echo "Passed:     $PASSED"
echo "Failed:     $FAILED"
echo "Success rate: $(echo "scale=1; $PASSED * 100 / $NUM_RUNS" | bc)%"
echo

if [ $FAILED -eq 0 ]; then
    echo "✓ All tests passed consistently!"
    exit 0
else
    echo "⚠ Some tests failed. Check logs in: $LOGDIR"
    echo "Failed runs:"
    for i in $(seq 1 $NUM_RUNS); do
        if ! grep -q "1 passed" "$LOGDIR/run-$i.log" 2>/dev/null; then
            echo "  - Run $i: $LOGDIR/run-$i.log"
        fi
    done
    exit 1
fi
