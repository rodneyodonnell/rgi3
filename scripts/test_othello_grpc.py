#!/usr/bin/env python3
"""
Test ELO progression for Othello with optional gRPC.

Usage:
    python scripts/test_othello_grpc.py              # Run once with gRPC
    python scripts/test_othello_grpc.py --no-grpc    # Run once without gRPC
    python scripts/test_othello_grpc.py --runs=10    # Run 10 times with gRPC
"""

import asyncio
import argparse
import sys
import time
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from rgi.rgizero.experiment import ExperimentConfig, ExperimentRunner
from rgi.rgizero.evaluators import ActionHistoryTransformerEvaluator, AsyncNetworkEvaluator
from rgi.rgizero.players.alphazero import AlphazeroPlayer
from rgi.rgizero.tournament import Tournament


async def run_othello_test(use_grpc: bool, num_games_per_gen: int, tournament_games: int = 500, run_id: int = 0):
    """Run single Othello ELO progression test."""

    config = ExperimentConfig(
        experiment_name=f"test-elo-othello-{run_id}",
        game_name="othello",
        num_generations=4,
        num_games_per_gen=num_games_per_gen,
        num_simulations=50,
        seed=42 + run_id,
        use_grpc=use_grpc,
    )

    # Minimal training args
    training_args = {
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 32,
        "n_max_context": 100,
        "dropout": 0.1,
        "bias": False,
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "max_iters": 5000,
        "max_epochs": 50,
        "learning_rate": 0.0005,
        "decay_lr": True,
        "min_lr": 0.00005,
        "lr_decay_iters": 5000,
        "warmup_iters": 100,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "dtype": "float32",
        "eval_iters": 20,
        "log_interval": 50,
        "eval_interval": 50,
        "early_stop_patience": 20,
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        runner = ExperimentRunner(
            config=config,
            base_dir=Path(temp_dir),
            training_args=training_args,
            progress_bar=False,
        )

        test_start = time.time()
        await runner.run_async()
        training_time = time.time() - test_start
        print(f"\n[Run {run_id}] Training time: {training_time:.1f}s ({training_time / 60:.1f} min)")

        # Load all models
        models = {gen_id: runner.load_model(gen_id) for gen_id in range(config.num_generations + 1)}

        # Run tournament
        @asynccontextmanager
        async def create_all_factories():
            evaluators = {}
            factories = {}

            for gen_id, model in models.items():
                serial_eval = ActionHistoryTransformerEvaluator(
                    model,
                    device=runner.device,
                    block_size=runner.n_max_context,
                    vocab=runner.action_vocab,
                    verbose=False,
                )
                async_eval = AsyncNetworkEvaluator(
                    base_evaluator=serial_eval,
                    max_batch_size=32,
                    verbose=False,
                )
                await async_eval.start()
                evaluators[gen_id] = async_eval

                def make_factory(evaluator):
                    def factory():
                        rng = np.random.default_rng(np.random.randint(0, 2**31))
                        return AlphazeroPlayer(
                            runner.game,
                            evaluator,
                            rng=rng,
                            add_noise=False,
                            simulations=20,
                        )

                    return factory

                factories[f"gen_{gen_id}"] = make_factory(async_eval)

            try:
                yield factories
            finally:
                for evaluator in evaluators.values():
                    await evaluator.stop()

        async with create_all_factories() as player_factories:
            tournament = Tournament(runner.game, player_factories, initial_elo=1000)

            # tournament_games is now a parameter
            tournament_start = time.time()
            await tournament.run(num_games=tournament_games, concurrent_games=10)
            tournament_time = time.time() - tournament_start

            elo_gen0 = tournament.stats["gen_0"].elo
            all_trained_elos = [tournament.stats[f"gen_{g}"].elo for g in range(1, config.num_generations + 1)]
            best_trained_elo = max(all_trained_elos)
            best_trained_gen = all_trained_elos.index(best_trained_elo) + 1

            elo_improvement = best_trained_elo - elo_gen0
            passed = elo_improvement >= 50

            total_time = training_time + tournament_time

            return {
                "run_id": run_id,
                "use_grpc": use_grpc,
                "num_games_per_gen": num_games_per_gen,
                "training_time": training_time,
                "tournament_time": tournament_time,
                "total_time": total_time,
                "gen0_elo": elo_gen0,
                "best_elo": best_trained_elo,
                "best_gen": best_trained_gen,
                "elo_improvement": elo_improvement,
                "passed": passed,
            }


async def main():
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Test Othello ELO progression")
    parser.add_argument("--no-grpc", action="store_true", help="Disable gRPC")
    parser.add_argument("--runs", type=int, default=1, help="Number of test runs")
    parser.add_argument("--games", type=int, default=500, help="Games per generation")
    parser.add_argument("--tournament-games", type=int, default=500, help="Total tournament games")
    args = parser.parse_args()

    use_grpc = not args.no_grpc

    print("=" * 60)
    print("Othello ELO Progression Test")
    print(f"gRPC: {use_grpc}, Runs: {args.runs}, Games/gen: {args.games}")
    print("=" * 60)

    results = []
    for i in range(args.runs):
        print(f"\n{'=' * 40}")
        print(f"Run {i + 1}/{args.runs}")
        print(f"{'=' * 40}")

        result = await run_othello_test(use_grpc, args.games, args.tournament_games, run_id=i)
        results.append(result)

        status = "✓ PASSED" if result["passed"] else "✗ FAILED"
        print(f"[Run {i}] {status}: ELO improvement = {result['elo_improvement']:.1f}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    avg_time = sum(r["total_time"] for r in results) / len(results)
    avg_improvement = sum(r["elo_improvement"] for r in results) / len(results)

    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print(f"Avg total time: {avg_time:.1f}s ({avg_time / 60:.1f} min)")
    print(f"Avg ELO improvement: {avg_improvement:.1f}")

    if failed > 0:
        print("\nFailed runs:")
        for r in results:
            if not r["passed"]:
                print(f"  Run {r['run_id']}: ELO improvement = {r['elo_improvement']:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
