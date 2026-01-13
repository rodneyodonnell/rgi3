#!/usr/bin/env python3
"""
Long-form training experiment to test model improvement across many generations.

This script trains a model for 20+ generations and runs tournaments to evaluate
ELO progression and validate that the training loop produces improving models.

Usage:
    python scripts/train_long_experiment.py --game count21 --generations 20
    python scripts/train_long_experiment.py --game connect4 --generations 15 --games-per-gen 200
"""

import asyncio
import argparse
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np

from rgi.rgizero.experiment import ExperimentConfig, ExperimentRunner
from rgi.rgizero.evaluators import ActionHistoryTransformerEvaluator, AsyncNetworkEvaluator
from rgi.rgizero.players.alphazero import AlphazeroPlayer
from rgi.rgizero.tournament import Tournament


DEFAULT_TRAINING_ARGS = {
    # Model architecture
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 64,
    "n_max_context": 200,
    "dropout": 0.0,
    "bias": False,
    # Training parameters
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "max_iters": 100000,
    "max_epochs": 20,
    "learning_rate": 0.003,
    "decay_lr": True,
    "min_lr": 0.0003,
    "lr_decay_iters": 2000,
    "warmup_iters": 50,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "dtype": "float32",
    "eval_iters": 50,
    "log_interval": 100,
    "eval_interval": 100,
    "early_stop_patience": 10,
}


async def run_tournament_evaluation(runner, generation_ids, num_games=50, concurrent_games=10):
    """
    Run a round-robin tournament between models from different generations.

    Args:
        runner: ExperimentRunner with trained models
        generation_ids: List of generation IDs to include in tournament
        num_games: Total number of games to play
        concurrent_games: Number of games to run concurrently

    Returns:
        Tournament object with results
    """
    # Load all models
    models = {gen_id: runner.load_model(gen_id) for gen_id in generation_ids}

    @asynccontextmanager
    async def create_all_factories():
        evaluators = {}
        factories = {}

        # Create evaluators for all models
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
                max_batch_size=64,
                verbose=False,
            )
            await async_eval.start()
            evaluators[gen_id] = async_eval

            # Create factory
            def make_factory(evaluator):
                def factory():
                    rng = np.random.default_rng(np.random.randint(0, 2**31))
                    return AlphazeroPlayer(
                        runner.game,
                        evaluator,
                        rng=rng,
                        add_noise=False,
                        simulations=50,
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
        await tournament.run(num_games=num_games, concurrent_games=concurrent_games)
        return tournament


async def main():
    parser = argparse.ArgumentParser(description="Run long-form training experiment")
    parser.add_argument(
        "--game", default="count21", choices=["count21", "connect4", "othello"], help="Game to train on"
    )
    parser.add_argument("--generations", type=int, default=20, help="Number of generations to train")
    parser.add_argument("--games-per-gen", type=int, default=150, help="Number of self-play games per generation")
    parser.add_argument("--simulations", type=int, default=50, help="MCTS simulations per move during self-play")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/long_run"), help="Output directory for experiment"
    )
    parser.add_argument("--tournament-interval", type=int, default=5, help="Run tournament every N generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create experiment config
    config = ExperimentConfig(
        experiment_name=f"long-{args.game}",
        game_name=args.game,
        num_generations=args.generations,
        num_games_per_gen=args.games_per_gen,
        num_simulations=args.simulations,
        seed=args.seed,
    )

    # Create runner
    runner = ExperimentRunner(
        config=config,
        base_dir=args.output_dir,
        training_args=DEFAULT_TRAINING_ARGS,
        progress_bar=True,
    )

    print("=" * 80)
    print(f"Long Training Experiment: {args.game.upper()}")
    print("=" * 80)
    print(f"Generations: {args.generations}")
    print(f"Games per generation: {args.games_per_gen}")
    print(f"MCTS simulations: {args.simulations}")
    print(f"Output directory: {args.output_dir}")
    print(f"Tournament every: {args.tournament_interval} generations")
    print("=" * 80)
    print()

    # Run training
    print("Starting training...")
    await runner.run_async()
    print(f"✓ Training completed for {args.generations} generations")
    print()

    # Run tournament evaluation
    print("=" * 80)
    print("TOURNAMENT EVALUATION")
    print("=" * 80)

    # Determine which generations to evaluate
    # Include: Gen 0 (random), every tournament_interval generations, and final generation
    eval_gens = set([0])  # Always include random baseline
    for gen in range(args.tournament_interval, args.generations + 1, args.tournament_interval):
        eval_gens.add(gen)
    eval_gens.add(args.generations)  # Always include final
    eval_gens = sorted(eval_gens)

    print(f"Evaluating generations: {eval_gens}")
    print()

    # Run tournament
    tournament = await run_tournament_evaluation(
        runner,
        eval_gens,
        num_games=len(eval_gens) * 20,  # Scale with number of players
        concurrent_games=10,
    )

    # Print results
    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    tournament.print_standings()

    # Analyze progression
    print()
    print("=" * 80)
    print("ELO PROGRESSION ANALYSIS")
    print("=" * 80)

    elo_by_gen = {}
    for gen_id in eval_gens:
        elo = tournament.stats[f"gen_{gen_id}"].elo
        games = tournament.stats[f"gen_{gen_id}"].games_played
        wins = tournament.stats[f"gen_{gen_id}"].wins
        elo_by_gen[gen_id] = elo

        indicator = ""
        if gen_id == 0:
            indicator = " (RANDOM BASELINE)"
        elif elo == max(elo_by_gen.values()):
            indicator = " ⭐ BEST MODEL"

        print(f"Gen {gen_id:2d}: ELO={elo:7.1f}, Games={games:3d}, Wins={wins:3d}{indicator}")

    # Check improvement
    gen0_elo = elo_by_gen[0]
    final_elo = elo_by_gen[args.generations]
    improvement = final_elo - gen0_elo

    print()
    print(f"Improvement from Gen 0 to Gen {args.generations}: {improvement:+.1f} ELO")

    if final_elo > gen0_elo:
        print(f"✓ Final model beats random baseline by {improvement:.1f} ELO")
    else:
        print(f"⚠ Final model is {-improvement:.1f} ELO worse than random - training may have failed")

    print()
    print(f"Results saved to: {args.output_dir / config.experiment_name}")


if __name__ == "__main__":
    asyncio.run(main())
