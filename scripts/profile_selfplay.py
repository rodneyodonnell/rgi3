#!/usr/bin/env python3
"""
Profile self-play to identify GPU vs CPU bottlenecks.

Usage:
    python scripts/profile_selfplay.py [--num-games 500] [--game othello]
"""

import argparse
import asyncio
import sys
import time
import threading
from pathlib import Path

import torch
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


class ResourceMonitor:
    """Monitor GPU and CPU usage during execution."""

    def __init__(self, device: str):
        self.device = device
        self.gpu_samples = []
        self.cpu_samples = []
        self.running = False
        self.thread = None

        # Check if we can monitor GPU
        self.can_monitor_gpu = device in ["cuda", "mps"]
        if device == "cuda":
            try:
                import pynvml
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.can_monitor_gpu = False
        elif device == "mps":
            # MPS doesn't have easy monitoring, we'll use torch.mps if available
            self.can_monitor_gpu = hasattr(torch.mps, "current_allocated_memory")

    def _monitor_loop(self):
        """Background thread to monitor resources."""
        import psutil

        # Also track per-core CPU usage
        self.cpu_per_core_samples = []

        while self.running:
            # CPU usage (overall)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_samples.append(cpu_percent)

            # Per-core CPU usage
            cpu_per_core = psutil.cpu_percent(interval=0, percpu=True)
            self.cpu_per_core_samples.append(cpu_per_core)

            # GPU usage
            if self.can_monitor_gpu:
                if self.device == "cuda":
                    try:
                        import pynvml
                        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                        self.gpu_samples.append(util.gpu)
                    except:
                        pass
                elif self.device == "mps":
                    # MPS monitoring is limited, just track if memory is allocated
                    if hasattr(torch.mps, "current_allocated_memory"):
                        allocated = torch.mps.current_allocated_memory()
                        # Normalize to 0-100 scale (rough estimate)
                        self.gpu_samples.append(min(100, allocated / (1024**3) * 10))

            time.sleep(0.1)

    def start(self):
        """Start monitoring."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring and return stats."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

        stats = {
            "cpu_mean": np.mean(self.cpu_samples) if self.cpu_samples else 0,
            "cpu_max": np.max(self.cpu_samples) if self.cpu_samples else 0,
        }

        if self.gpu_samples:
            stats["gpu_mean"] = np.mean(self.gpu_samples)
            stats["gpu_max"] = np.max(self.gpu_samples)

        # Per-core statistics
        if hasattr(self, 'cpu_per_core_samples') and self.cpu_per_core_samples:
            # Average across all samples for each core
            per_core_array = np.array(self.cpu_per_core_samples)
            stats["cpu_per_core_mean"] = np.mean(per_core_array, axis=0)
            stats["cpu_per_core_max"] = np.max(per_core_array, axis=0)
            stats["num_cores"] = len(stats["cpu_per_core_mean"])

        return stats


async def profile_selfplay(game_name: str, num_games: int, num_simulations: int = 50, max_time_seconds: int = None):
    """Profile self-play for the specified game."""

    print(f"Profiling {num_games} games of {game_name} with {num_simulations} MCTS simulations...")
    if max_time_seconds:
        print(f"(Will stop after {max_time_seconds} seconds for quick profiling)")
    print()

    # Setup
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    game = game_registry.create_game(game_name)
    action_vocab = Vocab(itos=[TOKENS.START_OF_GAME] + list(game.base_game.all_actions()))
    num_players = game.num_players(game.initial_state())

    # Model config (minimal for profiling)
    model_config = TransformerConfig(
        n_max_context=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
    )

    # Create random model
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
        verbose=True  # Enable verbose to see batching stats
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

    # Start resource monitoring
    monitor = ResourceMonitor(device)
    monitor.start()

    # Run self-play
    await async_evaluator.start()

    start_time = time.time()

    try:
        # Run games with semaphore
        limit = asyncio.Semaphore(1000)

        async def play_one_game():
            async with limit:
                player = player_factory()
                return await play_game_async(game, [player, player])

        tasks = [play_one_game() for _ in range(num_games)]

        from tqdm.asyncio import tqdm
        results = await tqdm.gather(*tasks, desc="Self-play")

    finally:
        await async_evaluator.stop()

    elapsed = time.time() - start_time

    # Stop monitoring
    resource_stats = monitor.stop()

    # Print results
    print()
    print("=" * 60)
    print("PROFILING RESULTS")
    print("=" * 60)
    print()
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Games played: {num_games}")
    print(f"Time per game: {elapsed/num_games:.2f}s")
    print(f"Games per second: {num_games/elapsed:.2f}")
    print()
    print(f"CPU utilization (overall):")
    print(f"  Mean: {resource_stats['cpu_mean']:.1f}%")
    print(f"  Max:  {resource_stats['cpu_max']:.1f}%")

    if "cpu_per_core_mean" in resource_stats:
        print()
        print(f"CPU per-core utilization ({resource_stats['num_cores']} cores):")
        for i, (mean, max_val) in enumerate(zip(resource_stats['cpu_per_core_mean'], resource_stats['cpu_per_core_max'])):
            print(f"  Core {i}: mean={mean:.1f}%, max={max_val:.1f}%")
        print()
        cores_above_50 = sum(1 for m in resource_stats['cpu_per_core_mean'] if m > 50)
        cores_above_80 = sum(1 for m in resource_stats['cpu_per_core_mean'] if m > 80)
        print(f"  Cores with >50% average utilization: {cores_above_50}/{resource_stats['num_cores']}")
        print(f"  Cores with >80% average utilization: {cores_above_80}/{resource_stats['num_cores']}")
    print()

    if "gpu_mean" in resource_stats:
        print(f"GPU utilization:")
        print(f"  Mean: {resource_stats['gpu_mean']:.1f}%")
        print(f"  Max:  {resource_stats['gpu_max']:.1f}%")
        print()

    # Evaluator stats
    print("Neural network inference:")
    print(f"  Total batches: {async_evaluator.stats['total_batches']}")
    print(f"  Total evaluations: {async_evaluator.stats['total_evals']}")
    print(f"  Mean batch size: {async_evaluator.stats['mean_batch_size']:.1f}")
    print(f"  Mean evals/sec: {async_evaluator.stats['mean_evals_per_sec']:.1f}")

    # Estimate time breakdown
    total_mcts_sims = num_games * 2 * num_simulations  # 2 players per game
    estimated_nn_time = async_evaluator.stats['total_evals'] / async_evaluator.stats['mean_evals_per_sec']
    mcts_time = elapsed - estimated_nn_time

    print()
    print("Estimated time breakdown:")
    print(f"  Neural network inference: {estimated_nn_time:.1f}s ({estimated_nn_time/elapsed*100:.1f}%)")
    print(f"  MCTS + game logic: {mcts_time:.1f}s ({mcts_time/elapsed*100:.1f}%)")
    print()
    print("Optimization opportunities:")
    if resource_stats['cpu_mean'] < 50:
        print("  ⚠ Low overall CPU utilization - could benefit from more parallelism")
        if "cpu_per_core_mean" in resource_stats:
            max_core = max(resource_stats['cpu_per_core_mean'])
            if max_core < 50:
                print(f"    → All cores underutilized (max core: {max_core:.1f}%) - increase concurrent games")
            elif max_core > 80:
                print(f"    → One core saturated ({max_core:.1f}%) - likely single-threaded bottleneck")
    if "gpu_mean" in resource_stats and resource_stats['gpu_mean'] < 30:
        print("  ⚠ Low GPU utilization - could benefit from larger batch sizes")
    if async_evaluator.stats['mean_batch_size'] < 100:
        print(f"  ⚠ Small batch size ({async_evaluator.stats['mean_batch_size']:.1f}) - consider tuning async batching")
    print()


def main():
    parser = argparse.ArgumentParser(description="Profile self-play performance")
    parser.add_argument("--num-games", type=int, default=500, help="Number of games to play")
    parser.add_argument("--game", type=str, default="othello", help="Game to profile")
    parser.add_argument("--simulations", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--max-time", type=int, default=None, help="Max time in seconds (for quick profiling)")

    args = parser.parse_args()

    asyncio.run(profile_selfplay(args.game, args.num_games, args.simulations, args.max_time))


if __name__ == "__main__":
    main()
