#!/usr/bin/env python3
"""
Specialized training script for Connect4 with a 60-minute time limit.
Includes periodic ELO tournament against Gen 0 baseline.
"""

import asyncio
import time
import argparse
from pathlib import Path
from contextlib import asynccontextmanager


from rgi.rgizero.experiment import ExperimentConfig, ExperimentRunner
from rgi.rgizero.evaluators import ActionHistoryTransformerEvaluator, AsyncNetworkEvaluator
from rgi.rgizero.players.alphazero import AlphazeroPlayer
from rgi.rgizero.tournament import Tournament

# Similar hyperparams to successful Othello runs
DEFAULT_TRAINING_ARGS = {
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 64,
    "n_max_context": 100,
    "dropout": 0.0,
    "bias": False,
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "max_iters": 3000,
    "max_epochs": 20,
    "learning_rate": 0.001,
    "decay_lr": True,
    "min_lr": 0.0001,
    "lr_decay_iters": 3000,
    "warmup_iters": 50,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "dtype": "float32",
    "eval_iters": 50,
    "log_interval": 100,
    "eval_interval": 100,
    "early_stop_patience": 5,
}


async def run_elo_check(runner, gen_id, num_games=20):
    """Run a quick tournament between Gen 0 and current generation."""
    print(f"\n--- ELO Check: Gen 0 vs Gen {gen_id} ---")

    gen0_model = runner.load_model(0)
    current_model = runner.load_model(gen_id)

    @asynccontextmanager
    async def create_factories():
        evaluators = []
        factories = {}

        for name, model in [("gen_0", gen0_model), (f"gen_{gen_id}", current_model)]:
            serial_eval = ActionHistoryTransformerEvaluator(
                model,
                device=runner.device,
                block_size=runner.n_max_context,
                vocab=runner.action_vocab,
                verbose=False,
            )
            async_eval = AsyncNetworkEvaluator(base_evaluator=serial_eval, max_batch_size=64, verbose=False)
            await async_eval.start()
            evaluators.append(async_eval)

            def make_factory(ev):
                def factory():
                    return AlphazeroPlayer(runner.game, ev, simulations=50)

                return factory

            factories[name] = make_factory(async_eval)

        try:
            yield factories
        finally:
            for ev in evaluators:
                await ev.stop()

    async with create_factories() as player_factories:
        tournament = Tournament(runner.game, player_factories, initial_elo=1000)
        await tournament.run(num_games=num_games, concurrent_games=10)
        tournament.print_standings()
        return tournament


async def main():
    parser = argparse.ArgumentParser(description="Train Connect4 with time limit")
    parser.add_argument("--limit-minutes", type=int, default=60, help="Stop after N minutes")
    parser.add_argument("--generations", type=int, default=50, help="Max generations")
    parser.add_argument("--games-per-gen", type=int, default=100, help="Games per generation")
    parser.add_argument("--simulations", type=int, default=50, help="Simulations per move")
    parser.add_argument("--elo-interval", type=int, default=5, help="Check ELO every N generations")

    args = parser.parse_args()

    start_time = time.time()
    time_limit_sec = args.limit_minutes * 60

    config = ExperimentConfig(
        experiment_name=f"c4_timed_{int(start_time)}",
        game_name="connect4",
        num_generations=args.generations,
        num_games_per_gen=args.games_per_gen,
        num_simulations=args.simulations,
    )

    runner = ExperimentRunner(
        config=config, base_dir=Path("experiments"), training_args=DEFAULT_TRAINING_ARGS, progress_bar=True
    )

    current_model = runner.initialize()

    for gen_id in range(1, args.generations + 1):
        elapsed = time.time() - start_time
        if elapsed > time_limit_sec:
            print(f"\nTime limit of {args.limit_minutes} minutes reached. Stopping training.")
            break

        print(f"\nTime remaining: {max(0, (time_limit_sec - elapsed) / 60):.1f} minutes")
        current_model = await runner.run_generation_step_async(gen_id, current_model)

        if gen_id % args.elo_interval == 0:
            await run_elo_check(runner, gen_id)

    final_elapsed = (time.time() - start_time) / 60
    print(f"\nTraining session finished in {final_elapsed:.1f} minutes.")
    print(f"Results saved to: {runner.exp_dir}")


if __name__ == "__main__":
    asyncio.run(main())
