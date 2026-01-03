#!/usr/bin/env python3
"""
Profile CPU hotspots in self-play using cProfile.

Usage:
    python scripts/profile_cpu_hotspots.py --num-games 50 --max-time 60
"""

import argparse
import asyncio
import cProfile
import pstats
import sys
from pathlib import Path
from io import StringIO

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rgi.rgizero.games import game_registry
from rgi.rgizero.models.transformer import TransformerConfig
from rgi.rgizero.models.action_history_transformer import ActionHistoryTransformer
from rgi.rgizero.models.tuner import create_random_model
from rgi.rgizero.data.trajectory_dataset import Vocab
from rgi.rgizero.common import TOKENS
from rgi.rgizero.players.alphazero import AlphazeroPlayer, play_game_async
from rgi.rgizero.evaluators import (
    ActionHistoryTransformerEvaluator,
    AsyncNetworkEvaluator,
)


async def profile_games(game_name: str, num_games: int, num_simulations: int = 50):
    """Run games for profiling."""

    # Setup
    device = "cpu"  # Use CPU for profiling to see CPU hotspots clearly
    print(f"Profiling {num_games} games of {game_name} on {device}...")

    game = game_registry.create_game(game_name)
    action_vocab = Vocab(itos=[TOKENS.START_OF_GAME] + list(game.base_game.all_actions()))
    num_players = game.num_players(game.initial_state())

    # Minimal model
    model_config = TransformerConfig(
        n_max_context=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
    )

    model = create_random_model(
        model_config,
        action_vocab.vocab_size,
        num_players,
        seed=42,
        device=device
    )

    # Setup evaluator
    serial_evaluator = ActionHistoryTransformerEvaluator(
        model, device=device, block_size=100, vocab=action_vocab
    )
    async_evaluator = AsyncNetworkEvaluator(
        base_evaluator=serial_evaluator,
        max_batch_size=1024,
        verbose=False
    )

    # Player factory
    master_rng = np.random.default_rng(42)

    def player_factory():
        seed = master_rng.integers(0, 2**31)
        rng = np.random.default_rng(seed)
        return AlphazeroPlayer(
            game,
            async_evaluator,
            rng=rng,
            add_noise=True,
            simulations=num_simulations,
        )

    # Run self-play
    await async_evaluator.start()

    try:
        limit = asyncio.Semaphore(1000)

        async def play_one_game():
            async with limit:
                player = player_factory()
                return await play_game_async(game, [player, player])

        tasks = [play_one_game() for _ in range(num_games)]
        results = await asyncio.gather(*tasks)

    finally:
        await async_evaluator.stop()

    return len(results)


def main():
    parser = argparse.ArgumentParser(description="Profile CPU hotspots in self-play")
    parser.add_argument("--num-games", type=int, default=50, help="Number of games")
    parser.add_argument("--game", type=str, default="othello", help="Game to profile")
    parser.add_argument("--simulations", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--max-time", type=int, default=60, help="Max profiling time in seconds")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"PROFILING CPU HOTSPOTS")
    print(f"{'='*60}\n")

    # Run with cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    # Run the async function
    games_completed = asyncio.run(profile_games(args.game, args.num_games, args.simulations))

    profiler.disable()

    print(f"\nCompleted {games_completed} games")
    print(f"\n{'='*60}")
    print(f"TOP CPU HOTSPOTS (by cumulative time)")
    print(f"{'='*60}\n")

    # Print stats
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(30)  # Top 30 functions

    print(s.getvalue())

    print(f"\n{'='*60}")
    print(f"TOP CPU HOTSPOTS (by time in function itself)")
    print(f"{'='*60}\n")

    s = StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.strip_dirs()
    stats.sort_stats('time')
    stats.print_stats(30)

    print(s.getvalue())

    # Look for specific bottlenecks
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION OPPORTUNITIES")
    print(f"{'='*60}\n")

    stats = pstats.Stats(profiler)

    # Find MCTS-related functions
    print("Functions containing 'mcts' or 'select' or 'expand':")
    stats.print_stats('mcts|select|expand|simulate|backprop')

    print("\nFunctions containing 'game' or 'legal' or 'apply':")
    stats.print_stats('game|legal|apply|clone')


if __name__ == "__main__":
    main()
