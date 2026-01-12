#!/usr/bin/env python3
"""
Overnight Model Experiments

Runs multiple experiments with different configurations, saves models,
and maintains an incremental ELO ranking as new models are trained.
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import asynccontextmanager

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from rgi.rgizero.experiment import ExperimentConfig, ExperimentRunner
from rgi.rgizero.evaluators import ActionHistoryTransformerEvaluator, AsyncNetworkEvaluator
from rgi.rgizero.models.action_history_transformer import ActionHistoryTransformer
from rgi.rgizero.models.transformer import TransformerConfig
from rgi.rgizero.players.alphazero import AlphazeroPlayer
from rgi.rgizero.tournament import Tournament


@dataclass
class ExperimentVariant:
    """Configuration for an experiment variant."""

    name: str
    description: str
    # Config overrides
    num_games_per_gen: int = 500
    num_simulations: int = 50
    num_generations: int = 4
    # Training overrides
    training_args: dict = field(default_factory=dict)
    # Time limit in seconds
    time_limit: int = 600  # 10 minutes


# Base training args (same as test fixture)
BASE_TRAINING_ARGS = {
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


# Define experiment variants
EXPERIMENTS = [
    ExperimentVariant(
        name="00_baseline",
        description="Baseline with default config",
    ),
    ExperimentVariant(
        name="01_higher_lr",
        description="Higher learning rate (0.001 vs 0.0005)",
        training_args={"learning_rate": 0.001, "min_lr": 0.0001},
    ),
    ExperimentVariant(
        name="02_more_sims",
        description="100 MCTS simulations (vs 50)",
        num_simulations=100,
        num_games_per_gen=300,  # Fewer games to compensate
    ),
    ExperimentVariant(
        name="03_bigger_model",
        description="Larger model: 4 layers, 64 embed",
        training_args={"n_layer": 4, "n_embd": 64, "n_head": 8},
        num_games_per_gen=400,  # Fewer games, slower training
    ),
    ExperimentVariant(
        name="04_more_gens",
        description="6 generations with fewer games each",
        num_generations=6,
        num_games_per_gen=300,
    ),
    ExperimentVariant(
        name="05_lower_dropout",
        description="Lower dropout (0.05 vs 0.1)",
        training_args={"dropout": 0.05},
    ),
    # Phase 2
    ExperimentVariant(
        name="06_combined",
        description="Phase 2: Bigger Model (4L/64E) + Lower Dropout (0.05)",
        training_args={"n_layer": 4, "n_embd": 64, "n_head": 8, "dropout": 0.05},
        num_games_per_gen=400,
    ),
    ExperimentVariant(
        name="07_even_bigger",
        description="Phase 2: Even Bigger (6L/64E) + Lower Dropout",
        training_args={"n_layer": 6, "n_embd": 64, "n_head": 8, "dropout": 0.05},
        num_games_per_gen=300,  # Reduced due to depth
    ),
    ExperimentVariant(
        name="08_bigger_more_sims",
        description="Phase 2: Bigger Model + 100 Sims",
        training_args={"n_layer": 4, "n_embd": 64, "n_head": 8, "dropout": 0.05},
        num_simulations=100,
        num_games_per_gen=200,  # Reduced drastically to compensate for sims & model size
    ),
    ExperimentVariant(
        name="09_huge_model",
        description="Phase 2: Huge Model (8L/128E)",
        training_args={"n_layer": 8, "n_embd": 128, "n_head": 8, "dropout": 0.05},
        num_games_per_gen=100,  # Very few games, purely exploratory
    ),
]


class ExperimentRunner2:
    """Runs experiments and maintains ELO pool."""

    def __init__(self, base_dir: Path, game_name: str = "othello"):
        self.base_dir = base_dir
        self.game_name = game_name
        self.model_pool = {}  # name -> model path
        self.elo_ratings = {}  # name -> ELO
        self.elo_log = []
        self.results = []

        # Load existing ELOs if available
        self._load_state()

    def _load_state(self):
        """Load previous state to enable resumption."""
        elo_path = self.base_dir / "elo_log.jsonl"
        if elo_path.exists():
            try:
                with open(elo_path, "r") as f:
                    lines = f.readlines()
                    if lines:
                        last_entry = json.loads(lines[-1])
                        # rankings is list of [name, elo]
                        self.elo_ratings = {name: elo for name, elo in last_entry["rankings"]}
                        self.log(f"Loaded {len(self.elo_ratings)} ELO ratings from history")
            except Exception as e:
                self.log(f"Failed to load ELO history: {e}")

    def log(self, msg: str):
        """Print with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")

    async def run_experiment(self, variant: ExperimentVariant) -> dict:
        """Run a single experiment variant."""
        self.log(f"Starting experiment: {variant.name}")
        self.log(f"  Description: {variant.description}")

        exp_dir = self.base_dir / variant.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Check if already done
        final_gen = variant.num_generations
        # ExperimentRunner creates a subdir with experiment_name inside base_dir
        # We passed exp_dir as base_dir, so it is exp_dir / variant.name
        runner_dir = exp_dir / variant.name
        model_path = runner_dir / "models" / f"gen-{final_gen}.pt"

        if model_path.exists():
            self.log(f"  Skipping {variant.name} (model exists)")
            self.model_pool[variant.name] = model_path
            return {
                "name": variant.name,
                "elapsed": 0,
                "model_path": str(model_path),
                "generations": final_gen,
                "skipped": True,
            }

        # Merge training args
        training_args = {**BASE_TRAINING_ARGS, **variant.training_args}

        # Save config
        config_data = {
            "name": variant.name,
            "description": variant.description,
            "num_games_per_gen": variant.num_games_per_gen,
            "num_simulations": variant.num_simulations,
            "num_generations": variant.num_generations,
            "training_args": training_args,
        }
        with open(exp_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        # Create experiment config
        config = ExperimentConfig(
            experiment_name=variant.name,
            game_name=self.game_name,
            num_generations=variant.num_generations,
            num_games_per_gen=variant.num_games_per_gen,
            num_simulations=variant.num_simulations,
            seed=42,
            use_grpc=False,  # Use local evaluator (simpler, avoids multiprocessing issues)
        )

        # Run experiment with time limit
        start_time = time.time()
        runner = ExperimentRunner(
            config=config,
            base_dir=exp_dir,
            training_args=training_args,
            progress_bar=False,
        )

        try:
            # Run training
            await runner.run_async()
            elapsed = time.time() - start_time
            self.log(f"  Completed in {elapsed:.1f}s")

            # Find final model
            final_gen = config.num_generations
            runner_dir = exp_dir / variant.name
            model_path = runner_dir / "models" / f"gen-{final_gen}.pt"

            if model_path.exists():
                self.model_pool[variant.name] = model_path
                self.log(f"  Saved model: {model_path}")
            else:
                self.log(f"  WARNING: Model not found at {model_path}")
                return {"name": variant.name, "error": "Model not found"}

            return {
                "name": variant.name,
                "elapsed": elapsed,
                "model_path": str(model_path),
                "generations": config.num_generations,
            }

        except Exception as e:
            self.log(f"  ERROR: {e}")
            return {"name": variant.name, "error": str(e)}

    async def run_incremental_elo(self, new_model_name: str, games_per_pair: int = 100):
        """Play new model against all existing models in pool."""
        if len(self.model_pool) < 2:
            # Initialize first model with baseline ELO
            self.elo_ratings[new_model_name] = 1000
            self.log(f"  Initial ELO for {new_model_name}: 1000")
            return

        self.log(f"  Running ELO games for {new_model_name}...")

        from rgi.rgizero.games import game_registry
        import torch

        game = game_registry.create_game(self.game_name)

        # Models will be loaded inside create_factories to ensure correct architecture

        # Create player factories
        @asynccontextmanager
        async def create_factories():
            evaluators = {}
            factories = {}

            for name, model_path in self.model_pool.items():
                # Load config to get correct architecture args
                exp_dir = self.base_dir / name
                with open(exp_dir / "config.json", "r") as f:
                    exp_config = json.load(f)

                # Get runner to build model with correct arch
                config = ExperimentConfig(
                    experiment_name=name,
                    game_name=self.game_name,
                    num_generations=1,
                    num_games_per_gen=1,
                    num_simulations=20,
                    seed=42,
                    use_grpc=False,
                )
                runner = ExperimentRunner(
                    config=config, base_dir=exp_dir, training_args=exp_config["training_args"], progress_bar=False
                )

                # Load weights and instantiate model
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

                # Determine config: prefer checkpoint config, fallback to runner config
                if isinstance(checkpoint, dict) and "model_config" in checkpoint:
                    conf = TransformerConfig(**checkpoint["model_config"])
                else:
                    conf = runner.model_config

                model = ActionHistoryTransformer(
                    config=conf, action_vocab_size=runner.action_vocab.vocab_size, num_players=runner.num_players
                )

                # Handle state_dict loading
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    model.load_state_dict(checkpoint["model"])
                elif isinstance(checkpoint, dict):
                    # Try to load dict as state_dict if it looks like one (and not a full checkpoint)
                    # But we already checked for 'model_config'.
                    # If it's a raw state dict, it won't have 'model_config'.
                    try:
                        model.load_state_dict(checkpoint)
                    except Exception:
                        # Maybe it IS a model object? (Unlikely with torch.save vs state_dict)
                        pass
                else:
                    # If checkpoint is the model itself
                    model = checkpoint

                # Ensure eval mode
                model.eval()

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
                evaluators[name] = async_eval

                def make_factory(evaluator, g=game):
                    def factory():
                        rng = np.random.default_rng(np.random.randint(0, 2**31))
                        return AlphazeroPlayer(g, evaluator, rng=rng, add_noise=False, simulations=20)

                    return factory

                factories[name] = make_factory(async_eval)

            try:
                yield factories
            finally:
                for evaluator in evaluators.values():
                    await evaluator.stop()

        # Run tournament
        async with create_factories() as player_factories:
            tournament = Tournament(game, player_factories, initial_elo=1000)

            # Restore previous ELOs
            for name, elo in self.elo_ratings.items():
                if name in tournament.stats:
                    tournament.stats[name].elo = elo

            # Play games
            total_games = len(self.model_pool) * games_per_pair
            await tournament.run(num_games=total_games, concurrent_games=10)

            # Update ELOs
            for name in self.model_pool:
                self.elo_ratings[name] = tournament.stats[name].elo

            # Log results
            sorted_elos = sorted(self.elo_ratings.items(), key=lambda x: -x[1])
            self.log("  Current ELO Rankings:")
            for i, (name, elo) in enumerate(sorted_elos):
                marker = " ← NEW" if name == new_model_name else ""
                self.log(f"    {i + 1}. {name}: {elo:.0f}{marker}")

            # Save to log
            self.elo_log.append(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "new_model": new_model_name,
                    "rankings": sorted_elos,
                }
            )

            with open(self.base_dir / "elo_log.jsonl", "a") as f:
                f.write(json.dumps(self.elo_log[-1]) + "\n")

    async def run_all(self, experiments: list[ExperimentVariant]):
        """Run all experiments with incremental ELO."""
        self.log(f"Starting {len(experiments)} experiments")
        self.log(f"Output directory: {self.base_dir}")

        for i, variant in enumerate(experiments):
            self.log(f"\n{'=' * 60}")
            self.log(f"Experiment {i + 1}/{len(experiments)}: {variant.name}")
            self.log(f"{'=' * 60}")

            # Run experiment
            result = await self.run_experiment(variant)
            self.results.append(result)

            if "error" not in result and not result.get("skipped", False):
                # Run ELO comparison
                await self.run_incremental_elo(variant.name)

            # Save intermediate results
            with open(self.base_dir / "results.json", "w") as f:
                json.dump(self.results, f, indent=2)

        # Final summary
        self.log(f"\n{'=' * 60}")
        self.log("FINAL RESULTS")
        self.log(f"{'=' * 60}")

        if self.elo_ratings:
            sorted_elos = sorted(self.elo_ratings.items(), key=lambda x: -x[1])
            self.log("\nFinal ELO Rankings:")
            for i, (name, elo) in enumerate(sorted_elos):
                self.log(f"  {i + 1}. {name}: {elo:.0f}")

            # Save final rankings
            with open(self.base_dir / "final_rankings.json", "w") as f:
                json.dump({"rankings": sorted_elos, "results": self.results}, f, indent=2)

        self.log("\nExperiments complete!")


async def main():
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    base_dir = Path("/Users/rodo/src/rgi3-claude/experiments/overnight_2026_01_08")

    runner = ExperimentRunner2(base_dir, game_name="othello")
    await runner.run_all(EXPERIMENTS)


if __name__ == "__main__":
    asyncio.run(main())
