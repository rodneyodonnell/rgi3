#!/usr/bin/env bash
# Run integration tests multiple times in parallel to verify consistency
#
# Usage:
#   ./scripts/test_consistency.sh                    # Run 10 parallel test runs (loss-based test)
#   ./scripts/test_consistency.sh 5                  # Run 5 parallel test runs
#   ./scripts/test_consistency.sh 10 elo             # Use ELO test instead

set -e

NUM_RUNS=${1:-10}
TEST_TYPE=${2:-"loss"}  # "loss" or "elo"

if [ "$TEST_TYPE" = "elo" ]; then
    TEST_NAME="test_elo_progression_across_generations"
    echo "Running $NUM_RUNS parallel ELO-based integration test runs..."
    echo "Note: ELO tests have higher variance due to small dataset."
else
    TEST_NAME="test_model_predictions_vs_training_data"
    echo "Running $NUM_RUNS parallel loss-based integration test runs..."
    echo "Loss-based tests are more reliable for consistency checking."
fi

echo "This helps verify the tests pass consistently and not just due to luck."
echo

# Create temp directory for logs
LOGDIR=$(mktemp -d)/consistency-tests
mkdir -p "$LOGDIR"

echo "Logs will be saved to: $LOGDIR"
echo

# Select test file based on test type
if [ "$TEST_TYPE" = "elo" ]; then
    TEST_FILE="tests/rgizero/test_integration.py::$TEST_NAME"
else
    TEST_FILE="tests/rgizero/test_model_predictions.py::$TEST_NAME"
fi

# Launch parallel test runs
for i in $(seq 1 $NUM_RUNS); do
    (
        echo "Starting run $i..."
        uv run pytest "$TEST_FILE" \
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
