#!/usr/bin/env python3
"""
Run ELO consistency tests for different games.

Usage:
    python scripts/test_elo_consistency.py count21 --runs 10
    python scripts/test_elo_consistency.py connect4 --runs 5
    python scripts/test_elo_consistency.py othello --runs 3
    python scripts/test_elo_consistency.py all --runs 3
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
import re


def run_test(game: str, run_number: int, log_dir: Path) -> tuple[bool, float | None]:
    """
    Run a single ELO test for the specified game.

    Returns:
        (passed, elo_improvement)
    """
    # Map game names to test names
    test_map = {
        "count21": "test_elo_progression_across_generations",
        "connect4": "test_elo_progression_connect4",
        "othello": "test_elo_progression_othello",
    }

    test_name = test_map.get(game)
    if not test_name:
        print(f"Unknown game: {game}")
        return False, None

    log_file = log_dir / f"{game}-run-{run_number}.log"

    cmd = [
        "uv", "run", "pytest",
        f"tests/rgizero/test_integration.py::{test_name}",
        "-v", "-s"
    ]

    print(f"Starting {game} run {run_number}...")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=7200  # 120 minute timeout for extended Othello experiments
        )

        # Write log
        log_file.write_text(result.stdout)

        # Check if passed
        passed = result.returncode == 0 and "1 passed" in result.stdout

        # Extract ELO improvement
        elo_improvement = None
        match = re.search(r"ELO Improvement: ([+-][0-9]+\.[0-9]+) ELO", result.stdout)
        if match:
            elo_improvement = float(match.group(1))

        if passed:
            print(f"✓ {game} run {run_number}: PASSED (Improvement: {elo_improvement:+.1f} ELO)" if elo_improvement else f"✓ {game} run {run_number}: PASSED")
        else:
            print(f"✗ {game} run {run_number}: FAILED")

        return passed, elo_improvement

    except subprocess.TimeoutExpired:
        print(f"✗ {game} run {run_number}: TIMEOUT")
        log_file.write_text("TEST TIMEOUT")
        return False, None
    except Exception as e:
        print(f"✗ {game} run {run_number}: ERROR - {e}")
        return False, None


def main():
    parser = argparse.ArgumentParser(description="Run ELO consistency tests")
    parser.add_argument(
        "game",
        choices=["count21", "connect4", "othello", "all"],
        help="Game to test (or 'all' for all games)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of test runs per game (default: 5)"
    )

    args = parser.parse_args()

    # Determine which games to test
    if args.game == "all":
        games = ["count21", "connect4", "othello"]
    else:
        games = [args.game]

    # Create temp directory for logs
    log_dir = Path(tempfile.mkdtemp()) / "elo-consistency-tests"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {args.runs} ELO consistency tests for: {', '.join(games)}")
    print(f"Logs will be saved to: {log_dir}")
    print()

    # Run tests for each game
    results = {}
    for game in games:
        print(f"\n{'=' * 60}")
        print(f"Testing {game.upper()}")
        print(f"{'=' * 60}\n")

        passed = 0
        failed = 0
        improvements = []

        for run_num in range(1, args.runs + 1):
            success, elo_improvement = run_test(game, run_num, log_dir)

            if success:
                passed += 1
                if elo_improvement is not None:
                    improvements.append(elo_improvement)
            else:
                failed += 1

        results[game] = {
            "passed": passed,
            "failed": failed,
            "improvements": improvements,
        }

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}\n")

    all_passed = True
    for game in games:
        r = results[game]
        total = r["passed"] + r["failed"]
        success_rate = (r["passed"] / total * 100) if total > 0 else 0

        print(f"{game.upper()}:")
        print(f"  Runs: {total}")
        print(f"  Passed: {r['passed']}")
        print(f"  Failed: {r['failed']}")
        print(f"  Success rate: {success_rate:.1f}%")

        if r["improvements"]:
            avg_improvement = sum(r["improvements"]) / len(r["improvements"])
            min_improvement = min(r["improvements"])
            max_improvement = max(r["improvements"])
            print(f"  ELO improvements:")
            print(f"    Average: {avg_improvement:+.1f}")
            print(f"    Range: {min_improvement:+.1f} to {max_improvement:+.1f}")

        print()

        if r["failed"] > 0:
            all_passed = False

    print(f"Logs saved to: {log_dir}")

    if all_passed:
        print("\n✓ All tests passed consistently!")
        return 0
    else:
        print("\n⚠ Some tests failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
