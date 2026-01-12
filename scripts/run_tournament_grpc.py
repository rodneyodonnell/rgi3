#!/usr/bin/env python3
"""
Demo script for running ELO tournament with gRPC-backed inference.

Shows multi-model tournament with ModelServerManager and GrpcEvaluator.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from rgi.rgizero.games import game_registry
from rgi.rgizero.data.trajectory_dataset import Vocab
from rgi.rgizero.common import TOKENS
from rgi.rgizero.models.transformer import TransformerConfig
from rgi.rgizero.models.tuner import create_random_model
from rgi.rgizero.players.alphazero import AlphazeroPlayer
from rgi.rgizero.serving import ModelServerManager, GrpcEvaluator
from rgi.rgizero.tournament import Tournament


def create_test_model(game, vocab, num_players, seed: int) -> str:
    """Create a random model and save to temp file."""
    config = TransformerConfig(
        n_max_context=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
    )
    model = create_random_model(config, vocab.vocab_size, num_players, seed=seed, device="cpu")

    # Save to temp file
    model_path = tempfile.mktemp(suffix=f"_model_{seed}.pt")
    torch.save({"model": model}, model_path)

    return model_path


async def main():
    print("=" * 60)
    print("gRPC ELO Tournament Demo")
    print("=" * 60)

    game_name = "othello"
    num_models = 3
    games_per_pair = 4  # Total ~12 games for 3 models
    concurrent_games = 8
    num_simulations = 10

    print(f"Game: {game_name}")
    print(f"Models: {num_models}")
    print(f"Concurrent games: {concurrent_games}")
    print()

    # Setup game and vocab
    game = game_registry.create_game(game_name)
    vocab = Vocab(itos=[TOKENS.START_OF_GAME] + list(game.base_game.all_actions()))
    num_players = game.num_players(game.initial_state())

    # Create test models
    print("Creating test models...")
    model_paths = [create_test_model(game, vocab, num_players, seed=i) for i in range(num_models)]
    for i, path in enumerate(model_paths):
        print(f"  Model_{i}: {path}")
    print()

    # Start model servers
    print("Starting inference servers...")
    with ModelServerManager(game_name=game_name, verbose=True) as manager:
        # Get ports for each model (this starts the servers)
        model_ports = {path: manager.get_port(path) for path in model_paths}
        print()

        # Create evaluators for each model
        evaluators = {}
        for path, port in model_ports.items():
            evaluator = GrpcEvaluator(
                host="localhost",
                port=port,
                vocab=vocab,
                vocab_size=vocab.vocab_size,
            )
            await evaluator.connect()
            evaluators[path] = evaluator

        # Create player factories
        def make_player_factory(evaluator):
            def factory():
                return AlphazeroPlayer(
                    game=game,
                    evaluator=evaluator,
                    rng=np.random.default_rng(),
                    add_noise=True,
                    simulations=num_simulations,
                )

            return factory

        player_factories = {f"Model_{i}": make_player_factory(evaluators[path]) for i, path in enumerate(model_paths)}

        # Run tournament
        print("Running tournament...")
        tournament = Tournament(game, player_factories, initial_elo=1200)
        total_games = num_models * (num_models - 1) * games_per_pair // 2  # Each pair plays games_per_pair times
        await tournament.run(num_games=total_games, concurrent_games=concurrent_games)

        # Print results
        tournament.print_standings()

        # Close evaluators
        for evaluator in evaluators.values():
            await evaluator.close()

    # Cleanup temp models
    for path in model_paths:
        Path(path).unlink(missing_ok=True)

    print("\nDone!")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn")
    asyncio.run(main())
