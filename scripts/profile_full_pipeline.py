#!/usr/bin/env python3
"""
Comprehensive profiler for the full RGIZero training pipeline.

Profiles self-play, training, and tournament phases with:
- Per-phase timing breakdown
- CPU utilization (overall and per-core)
- GPU utilization (MPS/CUDA)
- Bottleneck identification

Usage:
    uv run python scripts/profile_full_pipeline.py --game othello --num-games 100
    uv run python scripts/profile_full_pipeline.py --game all --num-games 50 --quick
"""

import argparse
import asyncio
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rgi.rgizero.common import TOKENS
from rgi.rgizero.data.trajectory_dataset import Vocab
from rgi.rgizero.evaluators import (
    ActionHistoryTransformerEvaluator,
    AsyncNetworkEvaluator,
)
from rgi.rgizero.experiment import ExperimentConfig, ExperimentRunner
from rgi.rgizero.games import game_registry
from rgi.rgizero.models.transformer import TransformerConfig
from rgi.rgizero.models.tuner import create_random_model
from rgi.rgizero.players.alphazero import AlphazeroPlayer, play_game_async
from rgi.rgizero.tournament import Tournament


@dataclass
class PhaseStats:
    """Statistics for a single phase of the pipeline."""

    name: str
    duration_seconds: float
    cpu_mean: float = 0.0
    cpu_max: float = 0.0
    cpu_per_core_mean: list = field(default_factory=list)
    cpu_per_core_max: list = field(default_factory=list)
    gpu_mean: float = 0.0
    gpu_max: float = 0.0
    # Additional phase-specific stats
    extra_stats: dict = field(default_factory=dict)

    def is_cpu_bound(self) -> bool:
        """Check if this phase appears CPU-bound."""
        # CPU-bound if CPU utilization is high and/or GPU is low
        return self.cpu_mean > 50 or (self.gpu_mean < 20 and self.cpu_mean > 20)

    def is_gpu_bound(self) -> bool:
        """Check if this phase appears GPU-bound."""
        return self.gpu_mean > 50

    def is_single_threaded_bottleneck(self) -> bool:
        """Check if we have a single-threaded CPU bottleneck."""
        if not self.cpu_per_core_mean:
            return False
        max_core = max(self.cpu_per_core_mean) if self.cpu_per_core_mean else 0
        avg_core = np.mean(self.cpu_per_core_mean) if self.cpu_per_core_mean else 0
        # Single-threaded if one core is much higher than average
        return max_core > 70 and avg_core < 30


@dataclass
class PipelineReport:
    """Full pipeline profiling report."""

    game_name: str
    device: str
    phases: list  # List of PhaseStats
    total_time: float = 0.0

    def get_time_breakdown(self) -> dict:
        """Get percentage breakdown of time by phase."""
        if self.total_time == 0:
            return {}
        return {phase.name: (phase.duration_seconds / self.total_time) * 100 for phase in self.phases}


class ResourceMonitor:
    """Monitor CPU and GPU usage during execution."""

    def __init__(self, device: str):
        self.device = device
        self.cpu_samples = []
        self.cpu_per_core_samples = []
        self.gpu_samples = []
        self.running = False
        self.thread = None

        # Check GPU monitoring capability
        self.can_monitor_gpu = device in ["cuda", "mps"]
        if device == "cuda":
            try:
                import pynvml

                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.can_monitor_gpu = False
        elif device == "mps":
            self.can_monitor_gpu = hasattr(torch.mps, "current_allocated_memory")

    def _monitor_loop(self):
        """Background monitoring thread."""
        while self.running:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_samples.append(cpu_percent)

            # Per-core CPU
            cpu_per_core = psutil.cpu_percent(interval=0, percpu=True)
            self.cpu_per_core_samples.append(cpu_per_core)

            # GPU usage
            if self.can_monitor_gpu:
                if self.device == "cuda":
                    try:
                        import pynvml

                        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                        self.gpu_samples.append(util.gpu)
                    except Exception:
                        pass
                elif self.device == "mps":
                    if hasattr(torch.mps, "current_allocated_memory"):
                        allocated = torch.mps.current_allocated_memory()
                        self.gpu_samples.append(min(100, allocated / (1024**3) * 10))

            time.sleep(0.1)

    def start(self):
        """Start monitoring."""
        self.cpu_samples = []
        self.cpu_per_core_samples = []
        self.gpu_samples = []
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict:
        """Stop monitoring and return stats."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

        stats = {
            "cpu_mean": np.mean(self.cpu_samples) if self.cpu_samples else 0,
            "cpu_max": np.max(self.cpu_samples) if self.cpu_samples else 0,
            "gpu_mean": np.mean(self.gpu_samples) if self.gpu_samples else 0,
            "gpu_max": np.max(self.gpu_samples) if self.gpu_samples else 0,
        }

        if self.cpu_per_core_samples:
            per_core_array = np.array(self.cpu_per_core_samples)
            stats["cpu_per_core_mean"] = list(np.mean(per_core_array, axis=0))
            stats["cpu_per_core_max"] = list(np.max(per_core_array, axis=0))

        return stats


class PipelineProfiler:
    """Profiles the full RGIZero training pipeline."""

    def __init__(
        self,
        game_name: str,
        num_games: int = 100,
        num_simulations: int = 50,
        num_generations: int = 2,
        tournament_games: int = 100,
        training_args: Optional[dict] = None,
    ):
        self.game_name = game_name
        self.num_games = num_games
        self.num_simulations = num_simulations
        self.num_generations = num_generations
        self.tournament_games = tournament_games

        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        # Default minimal training args - tuned for small profiling datasets
        self.training_args = training_args or {
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 32,
            "n_max_context": 100,
            "dropout": 0.0,
            "bias": False,
            "batch_size": 16,  # Smaller batch for small datasets
            "gradient_accumulation_steps": 1,  # Critical for small datasets
            "max_epochs": 2,  # Just a couple epochs for profiling
            "max_iters": 50,  # Limit iterations
            "learning_rate": 0.001,
            "decay_lr": False,
            "dtype": "float32",
            "eval_iters": 5,
            "log_interval": 25,
            "eval_interval": 25,
            "early_stop_patience": 3,
        }

        # Game setup
        self.game = game_registry.create_game(game_name)
        self.action_vocab = Vocab(itos=[TOKENS.START_OF_GAME] + list(self.game.base_game.all_actions()))
        self.num_players = self.game.num_players(self.game.initial_state())

        # Monitor
        self.monitor = ResourceMonitor(self.device)

    async def profile_selfplay(self) -> PhaseStats:
        """Profile the self-play phase."""
        print(f"\n{'='*60}")
        print(f"PROFILING SELF-PLAY: {self.game_name}")
        print(f"Games: {self.num_games}, Simulations: {self.num_simulations}")
        print(f"{'='*60}\n")

        # Create random model
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
            self.action_vocab.vocab_size,
            self.num_players,
            seed=42,
            device=self.device,
        )

        # Setup evaluator
        serial_evaluator = ActionHistoryTransformerEvaluator(
            model, device=self.device, block_size=100, vocab=self.action_vocab
        )
        async_evaluator = AsyncNetworkEvaluator(base_evaluator=serial_evaluator, max_batch_size=1024, verbose=False)

        # Player factory
        master_rng = np.random.default_rng(42)

        def player_factory():
            seed = master_rng.integers(0, 2**31)
            rng = np.random.default_rng(seed)
            return AlphazeroPlayer(
                self.game,
                async_evaluator,
                rng=rng,
                add_noise=True,
                simulations=self.num_simulations,
            )

        # Start monitoring and run
        self.monitor.start()
        start_time = time.time()

        await async_evaluator.start()
        try:
            limit = asyncio.Semaphore(1000)

            async def play_one_game():
                async with limit:
                    player = player_factory()
                    return await play_game_async(self.game, [player, player])

            from tqdm.asyncio import tqdm

            tasks = [play_one_game() for _ in range(self.num_games)]
            results = await tqdm.gather(*tasks, desc="Self-play")
        finally:
            await async_evaluator.stop()

        duration = time.time() - start_time
        stats = self.monitor.stop()

        # Get stats from the base evaluator
        base_stats = serial_evaluator
        extra = {
            "games_played": len(results),
            "games_per_second": len(results) / duration,
            "total_batches": getattr(base_stats, "total_batches", 0),
            "total_evals": getattr(base_stats, "total_evals", 0),
            "mean_batch_size": (
                base_stats.total_evals / base_stats.total_batches if base_stats.total_batches > 0 else 0
            ),
            "mean_evals_per_sec": base_stats.total_evals / base_stats.total_time if base_stats.total_time > 0 else 0,
        }

        return PhaseStats(
            name="Self-Play",
            duration_seconds=duration,
            cpu_mean=stats["cpu_mean"],
            cpu_max=stats["cpu_max"],
            cpu_per_core_mean=stats.get("cpu_per_core_mean", []),
            cpu_per_core_max=stats.get("cpu_per_core_max", []),
            gpu_mean=stats["gpu_mean"],
            gpu_max=stats["gpu_max"],
            extra_stats=extra,
        )

    async def profile_training(self) -> PhaseStats:
        """Profile the training phase using a full experiment run."""
        print(f"\n{'='*60}")
        print(f"PROFILING TRAINING: {self.game_name}")
        print(f"Games/gen: {self.num_games}, Generations: {self.num_generations}")
        print(f"{'='*60}\n")

        # Create temp directory for experiment
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name=f"profile-{self.game_name}",
                game_name=self.game_name,
                num_generations=self.num_generations,
                num_games_per_gen=self.num_games,
                num_simulations=self.num_simulations,
                seed=42,
            )

            runner = ExperimentRunner(
                config=config,
                base_dir=Path(temp_dir),
                training_args=self.training_args,
                progress_bar=True,
            )

            # Initialize
            model = runner.initialize()

            # Track time for training separately from self-play
            total_training_time = 0.0
            training_cpu_samples = []

            for gen_id in range(1, self.num_generations + 1):
                print(f"\n--- Generation {gen_id} ---")

                # Self-play (we're measuring training, so just record but don't attribute)
                await runner.play_generation_async(model, gen_id, write_dataset=True)

                # Profile training specifically
                self.monitor.start()
                train_start = time.time()

                model = runner.train_generation(model, gen_id)

                train_duration = time.time() - train_start
                total_training_time += train_duration
                stats = self.monitor.stop()
                training_cpu_samples.append(stats)

                print(f"  Training time: {train_duration:.1f}s")

        # Aggregate training stats
        if training_cpu_samples:
            cpu_means = [s["cpu_mean"] for s in training_cpu_samples]
            cpu_maxes = [s["cpu_max"] for s in training_cpu_samples]
            gpu_means = [s["gpu_mean"] for s in training_cpu_samples]
            gpu_maxes = [s["gpu_max"] for s in training_cpu_samples]

            return PhaseStats(
                name="Training",
                duration_seconds=total_training_time,
                cpu_mean=np.mean(cpu_means),
                cpu_max=max(cpu_maxes),
                gpu_mean=np.mean(gpu_means),
                gpu_max=max(gpu_maxes),
                extra_stats={
                    "generations": self.num_generations,
                    "avg_time_per_gen": total_training_time / self.num_generations,
                },
            )

        return PhaseStats(name="Training", duration_seconds=total_training_time)

    async def profile_tournament(self) -> PhaseStats:
        """Profile the tournament/ELO calculation phase."""
        print(f"\n{'='*60}")
        print(f"PROFILING TOURNAMENT: {self.game_name}")
        print(f"Games: {self.tournament_games}")
        print(f"{'='*60}\n")

        # Create two random models to pit against each other
        model_config = TransformerConfig(
            n_max_context=100,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.0,
            bias=False,
        )

        model1 = create_random_model(
            model_config, self.action_vocab.vocab_size, self.num_players, seed=42, device=self.device
        )
        model2 = create_random_model(
            model_config, self.action_vocab.vocab_size, self.num_players, seed=123, device=self.device
        )

        # Setup evaluators
        evaluators = []
        factories = {}

        for i, model in enumerate([model1, model2]):
            serial_eval = ActionHistoryTransformerEvaluator(
                model, device=self.device, block_size=100, vocab=self.action_vocab
            )
            async_eval = AsyncNetworkEvaluator(base_evaluator=serial_eval, max_batch_size=32, verbose=False)
            evaluators.append(async_eval)

            def make_factory(evaluator):
                def factory():
                    rng = np.random.default_rng(np.random.randint(0, 2**31))
                    return AlphazeroPlayer(self.game, evaluator, rng=rng, add_noise=False, simulations=20)

                return factory

            factories[f"model_{i}"] = make_factory(async_eval)

        # Start evaluators
        for e in evaluators:
            await e.start()

        try:
            tournament = Tournament(self.game, factories, initial_elo=1000)

            self.monitor.start()
            start_time = time.time()

            await tournament.run(num_games=self.tournament_games, concurrent_games=10)

            duration = time.time() - start_time
            stats = self.monitor.stop()

        finally:
            for e in evaluators:
                await e.stop()

        return PhaseStats(
            name="Tournament",
            duration_seconds=duration,
            cpu_mean=stats["cpu_mean"],
            cpu_max=stats["cpu_max"],
            cpu_per_core_mean=stats.get("cpu_per_core_mean", []),
            cpu_per_core_max=stats.get("cpu_per_core_max", []),
            gpu_mean=stats["gpu_mean"],
            gpu_max=stats["gpu_max"],
            extra_stats={
                "games": self.tournament_games,
                "games_per_second": self.tournament_games / duration,
            },
        )

    async def run_full_profile(self) -> PipelineReport:
        """Run complete pipeline profiling."""
        print(f"\n{'#'*60}")
        print(f"FULL PIPELINE PROFILING: {self.game_name.upper()}")
        print(f"Device: {self.device}")
        print(f"{'#'*60}")

        total_start = time.time()

        phases = []

        # Profile self-play
        selfplay_stats = await self.profile_selfplay()
        phases.append(selfplay_stats)

        # Profile training (includes self-play internally but we measure training time)
        training_stats = await self.profile_training()
        phases.append(training_stats)

        # Profile tournament
        tournament_stats = await self.profile_tournament()
        phases.append(tournament_stats)

        total_time = time.time() - total_start

        return PipelineReport(
            game_name=self.game_name,
            device=self.device,
            phases=phases,
            total_time=total_time,
        )


def format_report(report: PipelineReport) -> str:
    """Format a pipeline report as markdown."""
    output = StringIO()

    output.write(f"# Performance Profile: {report.game_name.title()}\n\n")
    output.write(f"**Device:** {report.device}\n")
    output.write(f"**Total Time:** {report.total_time:.1f}s ({report.total_time/60:.1f} min)\n\n")

    # Time breakdown
    output.write("## Time Breakdown\n\n")
    output.write("| Phase | Time (s) | % of Total |\n")
    output.write("|-------|----------|------------|\n")
    for phase in report.phases:
        pct = (phase.duration_seconds / report.total_time) * 100 if report.total_time > 0 else 0
        output.write(f"| {phase.name} | {phase.duration_seconds:.1f} | {pct:.1f}% |\n")
    output.write("\n")

    # Resource utilization
    output.write("## Resource Utilization\n\n")
    output.write("| Phase | CPU Mean | CPU Max | GPU Mean | GPU Max |\n")
    output.write("|-------|----------|---------|----------|----------|\n")
    for phase in report.phases:
        output.write(
            f"| {phase.name} | {phase.cpu_mean:.1f}% | {phase.cpu_max:.1f}% | "
            f"{phase.gpu_mean:.1f}% | {phase.gpu_max:.1f}% |\n"
        )
    output.write("\n")

    # Bottleneck analysis
    output.write("## Bottleneck Analysis\n\n")
    for phase in report.phases:
        output.write(f"### {phase.name}\n\n")
        if phase.is_single_threaded_bottleneck():
            output.write("⚠️ **Single-threaded CPU bottleneck detected**\n")
            if phase.cpu_per_core_mean:
                max_core = max(phase.cpu_per_core_mean)
                output.write(f"- One core at {max_core:.1f}% while others are idle\n")
            output.write("- **Recommendation:** Multiprocessing would help\n\n")
        elif phase.is_cpu_bound():
            output.write("🔥 **CPU-bound phase**\n")
            output.write(f"- CPU utilization: {phase.cpu_mean:.1f}% average\n")
            output.write("- **Recommendation:** More CPU cores or optimization would help\n\n")
        elif phase.is_gpu_bound():
            output.write("🎮 **GPU-bound phase**\n")
            output.write(f"- GPU utilization: {phase.gpu_mean:.1f}% average\n")
            output.write("- **Recommendation:** Larger batch sizes or faster GPU would help\n\n")
        else:
            output.write("✅ **Balanced** - neither CPU nor GPU saturated\n\n")

        # Extra stats
        if phase.extra_stats:
            output.write("**Phase-specific stats:**\n")
            for key, value in phase.extra_stats.items():
                if isinstance(value, float):
                    output.write(f"- {key}: {value:.2f}\n")
                else:
                    output.write(f"- {key}: {value}\n")
            output.write("\n")

    # Per-core breakdown for self-play (if available)
    selfplay_phase = next((p for p in report.phases if p.name == "Self-Play"), None)
    if selfplay_phase and selfplay_phase.cpu_per_core_mean:
        output.write("## Per-Core CPU Utilization (Self-Play)\n\n")
        output.write("| Core | Mean % | Max % |\n")
        output.write("|------|--------|-------|\n")
        for i, (mean, max_val) in enumerate(
            zip(selfplay_phase.cpu_per_core_mean, selfplay_phase.cpu_per_core_max, strict=False)
        ):
            output.write(f"| {i} | {mean:.1f} | {max_val:.1f} |\n")
        output.write("\n")

        cores_above_50 = sum(1 for m in selfplay_phase.cpu_per_core_mean if m > 50)
        total_cores = len(selfplay_phase.cpu_per_core_mean)
        output.write(f"**Cores with >50% utilization:** {cores_above_50}/{total_cores}\n\n")

    return output.getvalue()


async def main():
    parser = argparse.ArgumentParser(description="Profile full RGIZero training pipeline")
    parser.add_argument("--game", type=str, default="othello", help="Game to profile (count21, connect4, othello, all)")
    parser.add_argument("--num-games", type=int, default=100, help="Number of self-play games per phase")
    parser.add_argument("--num-simulations", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--num-generations", type=int, default=2, help="Training generations")
    parser.add_argument("--tournament-games", type=int, default=100, help="Tournament games for ELO phase")
    parser.add_argument("--output", type=str, default=None, help="Output file for markdown report")
    parser.add_argument("--quick", action="store_true", help="Quick mode with reduced params")

    args = parser.parse_args()

    if args.quick:
        args.num_games = 50
        args.num_simulations = 30
        args.num_generations = 1
        args.tournament_games = 50

    games = ["count21", "connect4", "othello"] if args.game == "all" else [args.game]

    all_reports = []

    for game_name in games:
        profiler = PipelineProfiler(
            game_name=game_name,
            num_games=args.num_games,
            num_simulations=args.num_simulations,
            num_generations=args.num_generations,
            tournament_games=args.tournament_games,
        )

        report = await profiler.run_full_profile()
        all_reports.append(report)

        # Print formatted report
        formatted = format_report(report)
        print("\n" + formatted)

    # Write to file if specified
    if args.output:
        with open(args.output, "w") as f:
            f.write("# RGIZero Pipeline Performance Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for report in all_reports:
                f.write(format_report(report))
                f.write("\n---\n\n")
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
