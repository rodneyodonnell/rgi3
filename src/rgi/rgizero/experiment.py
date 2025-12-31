import os
import json
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import torch
import numpy as np

from rgi.rgizero.games import game_registry
from rgi.rgizero.models.transformer import TransformerConfig
from rgi.rgizero.models.action_history_transformer import ActionHistoryTransformer
from rgi.rgizero.models.tuner import create_random_model, train_model
from rgi.rgizero.train import TrainConfig
from rgi.rgizero.data.trajectory_dataset import TrajectoryDatasetBuilder, Vocab, build_trajectory_loader
from rgi.rgizero.common import TOKENS
from rgi.rgizero.players.alphazero import AlphazeroPlayer, play_game_async
from rgi.rgizero.evaluators import (
    ActionHistoryTransformerEvaluator,
    AsyncNetworkEvaluator,
)

import asyncio


@dataclass
class ExperimentConfig:
    experiment_name: str
    game_name: str
    num_generations: int
    num_games_per_gen: int
    num_simulations: int
    # model_size: str  # "tiny", "small", etc.
    # train_batch_size: int = 2048
    # max_training_epochs: int = 10
    parent_experiment_name: Optional[str] = None
    parent_generation_cap: Optional[int] = None
    # gradient_accumulation_steps: int = 1
    seed: int = 42
    training_window_size: Optional[int] = 10

    def to_json(self):
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(data: dict):
        return ExperimentConfig(**data)


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, base_dir: Path, training_args: Optional[dict] = None, progress_bar=True):
        self.config = config
        self.base_dir = base_dir
        self.exp_dir = base_dir / config.experiment_name
        self.data_dir = self.exp_dir / "data"
        self.models_dir = self.exp_dir / "models"
        self.progress_bar = progress_bar

        # Parent directories for forking
        self.parent_data_dir: Optional[Path] = None
        self.parent_models_dir: Optional[Path] = None
        if config.parent_experiment_name:
            parent_exp_dir = base_dir / config.parent_experiment_name
            self.parent_data_dir = parent_exp_dir / "data"
            self.parent_models_dir = parent_exp_dir / "models"

        if training_args is None:
            training_args = {}

        # Setup directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        self._save_config()

        # Initialize Game & Vocab
        # Note: We create a throwaway game instance just to get vocab/config info.
        self.game = game_registry.create_game(config.game_name)
        self.action_vocab = Vocab(itos=[TOKENS.START_OF_GAME] + list(self.game.base_game.all_actions()))
        self.num_players = self.game.num_players(self.game.initial_state())

        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        # Model Configuration using the Factory defaults roughly based on size
        # TODO: Move this configuration to central place or config object?
        n_max_context = 7 * 6 + 2  # Connect4 specific...
        # TODO: Make n_max_context dynamic based on game
        if config.game_name == "connect4":
            n_max_context = 7 * 6 + 2
        else:
            n_max_context = 100  # Fallback

        self.n_max_context = n_max_context

        # configs = {
        #     "tiny": TransformerConfig(n_max_context=n_max_context, n_layer=2, n_head=2, n_embd=8),
        #     "small": TransformerConfig(n_max_context=n_max_context, n_layer=4, n_head=4, n_embd=32),
        #     "large": TransformerConfig(n_max_context=n_max_context, n_layer=8, n_head=8, n_embd=128),
        # }
        # self.model_config = configs.get(config.model_size, configs["small"])
        model_config_keys = {f.name for f in dataclasses.fields(TransformerConfig)}
        model_config_dict = {k: v for k, v in training_args.items() if k in model_config_keys}
        self.model_config = TransformerConfig(**model_config_dict)

        train_config_keys = {f.name for f in dataclasses.fields(TrainConfig)}
        self.train_config_dict = {k: v for k, v in training_args.items() if k in train_config_keys}

        unused_keys = {k: v for k, v in training_args.items() if k not in model_config_keys and k not in self.train_config_dict}
        if unused_keys:
            raise ValueError(f"Unused training args: {unused_keys}")

    def _save_config(self):
        with open(self.exp_dir / "config.json", "w") as f:
            json.dump(self.config.to_json(), f, indent=2)

    def get_trajectory_paths(self, gen_id: int) -> list[Path]:
        start_gen = 1
        if self.config.training_window_size is not None:
             start_gen = max(1, gen_id - self.config.training_window_size + 1)
        
        paths = [self.get_trajectory_path(i) for i in range(start_gen, gen_id + 1)]
        return paths

    def get_trajectory_path(self, gen_id: int) -> Path:
        """Get path for trajectory data, handling overlay/forking logic."""
        filename = f"gen-{gen_id}"
        local_path = self.data_dir / filename

        if local_path.exists():
            return local_path

        # Check parent if eligible
        if self.parent_data_dir and (self.config.parent_generation_cap is None or gen_id <= self.config.parent_generation_cap):
            parent_path = self.parent_data_dir / filename
            if parent_path.exists():
                print(f"Using forked data for gen {gen_id} from {parent_path}")
                return parent_path

        return local_path  # Default to local (even if not exists, for writing)

    def get_model_path(self, gen_id: int) -> Path:
        """Get path for model checkpoint, handling overlay/forking logic."""
        filename = f"gen-{gen_id}.pt"
        local_path = self.models_dir / filename

        if local_path.exists():
            return local_path

        if self.parent_models_dir and (self.config.parent_generation_cap is None or gen_id <= self.config.parent_generation_cap):
            parent_path = self.parent_models_dir / filename
            if parent_path.exists():
                print(f"Using forked model for gen {gen_id} from {parent_path}")
                return parent_path

        return local_path

    def load_model(self, gen_id: int) -> ActionHistoryTransformer:
        path = self.get_model_path(gen_id)
        if not path.exists():
            raise FileNotFoundError(f"Model for gen {gen_id} not found at {path}")

        checkpoint = torch.load(path, map_location=self.device)
        # Handle cases where model config might be in the checkpoint vs self.model_config
        # For now, we trust the checkpoint's config if present, else self.model_config

        conf = checkpoint.get("model_config")
        if conf:
            config = TransformerConfig(**conf)
        else:
            config = self.model_config

        model = ActionHistoryTransformer(
            config=config, action_vocab_size=self.action_vocab.vocab_size, num_players=self.num_players
        )
        model.load_state_dict(checkpoint["model"])
        model.to(self.device)
        return model

    def save_model(self, model, gen_id: int, trainer_stats: dict):
        path = self.models_dir / f"gen-{gen_id}.pt"
        checkpoint = {
            "model": model.state_dict(),
            "model_config": dataclasses.asdict(model.config),
            "vocab": self.action_vocab.to_dict(),
            "trainer_stats": trainer_stats,
            "num_players": self.num_players,
        }
        torch.save(checkpoint, path)
        print(f"Saved model to {path}")

    def initialize(self) -> ActionHistoryTransformer:
        """Initialize Generation 0 (Random) if needed."""
        print(f"Starting Experiment: {self.config.experiment_name}")

        # Generation 0: Random Initialization
        model = create_random_model(
            self.model_config, self.action_vocab.vocab_size, self.num_players, seed=self.config.seed, device=self.device
        )

        # Check if Gen 0 is already saved (e.g. if we restart)
        gen0_path = self.get_model_path(0)
        if gen0_path.exists():
            print("Loading existing Gen 0 model.")
            model = self.load_model(0)
        else:
            print("Initializing Random Gen 0 model.")
            self.save_model(model, 0, {"description": "random init"})

        return model

    async def run_generation_step_async(
        self, gen_id: int, current_model: ActionHistoryTransformer
    ) -> ActionHistoryTransformer:
        """Run a single generation step: Self-Play -> Train. Returns updated model."""
        print(f"\n=== Generation {gen_id} ===")

        # 1. Self Play
        dataset_path = self.get_trajectory_path(gen_id)
        if dataset_path.exists():
            print(f"Dataset for gen {gen_id} exists at {dataset_path}. Skipping play.")
        else:
            print(f"Playing {self.config.num_games_per_gen} games...")
            await self.play_generation_async(current_model, gen_id)
            # Ensure it exists now
            dataset_path = self.get_trajectory_path(gen_id)

        # 2. Training
        model_path = self.get_model_path(gen_id)
        if model_path.exists():
            print(f"Model for gen {gen_id} exists at {model_path}. Loading.")
            updated_model = self.load_model(gen_id)
        else:
            print(f"Training model for gen {gen_id}...")
            updated_model = self.train_generation(current_model, gen_id)

        return updated_model

    async def run_async(self):
        current_model = self.initialize()

        for gen_id in range(1, self.config.num_generations + 1):
            current_model = await self.run_generation_step_async(gen_id, current_model)

    async def play_generation_async(self, model, gen_id, write_dataset=True):
        """Run self-play and save trajectory dataset."""
        # Setup Evaluator
        serial_evaluator = ActionHistoryTransformerEvaluator(
            model, device=self.device, block_size=self.n_max_context, vocab=self.action_vocab
        )
        async_evaluator = AsyncNetworkEvaluator(base_evaluator=serial_evaluator, max_batch_size=1024, verbose=False)

        # Player Factory
        # We need a closure to act as the factory for play_game_async loop
        master_rng = np.random.default_rng(self.config.seed + gen_id)

        def player_factory():
            seed = master_rng.integers(0, 2**31)
            rng = np.random.default_rng(seed)
            return AlphazeroPlayer(
                self.game,
                async_evaluator,
                rng=rng,
                add_noise=True,  # Exploration noise!
                simulations=self.config.num_simulations,
            )

        # Run Games
        await async_evaluator.start()
        try:
            results = await self._play_games_async(player_factory)
        finally:
            await async_evaluator.stop()

        # Write Dataset
        if write_dataset:
            print(f"Writing {len(results)} trajectories...")
            self._write_dataset(results, gen_id)

    async def _play_games_async(self, player_factory):
        """Helper to run games in parallel."""
        # Using a semaphore to limit concurrency if needed, though AsyncEvaluator handles batching
        # Actually, we rely on the AsyncNetworkEvaluator's max_batch_size to cap practical throughput,
        # but we shouldn't spawn infinite tasks.

        limit = asyncio.Semaphore(1000)  # Max concurrent games

        async def secure_semaphore_and_play_game_async():
            async with limit:
                player = player_factory()
                # Self-play: same player instance (or identical clones) for both sides usually fine for AlphaZero
                return await play_game_async(self.game, [player, player])

        tasks = [secure_semaphore_and_play_game_async() for _ in range(self.config.num_games_per_gen)]
        
        if self.progress_bar:
            from tqdm.asyncio import tqdm
            # TODO: tqdm.gather fails while stepping in the debugger? not sure why?
            results = await tqdm.gather(*tasks, desc="Self Play")
        else:
            results = await asyncio.gather(*tasks)
        return results

    def _write_dataset(self, results, gen_id):
        builder = TrajectoryDatasetBuilder(self.action_vocab)

        # Helper from notebook
        action_idx_to_vocab_idx = self.action_vocab.encode(list(self.game.base_game.all_actions()))

        for res in results:
            action_history = res["action_history"]
            length = len(action_history)
            legal_policies = res["legal_policies"]
            legal_action_idx = res["legal_action_idx"]  # these are indices into game.all_actions()
            rewards = res["rewards"]

            fixed_width_policies = np.zeros((length, self.action_vocab.vocab_size))
            for i in range(length):
                # Mapping the sparse legal policies to the full vocab width
                # legal_action_idx[i] is a list of indices into the game's static action list
                # We need to map those to the vocab indices

                # Note: This logic assumes legal_action_idx matches indices in game.all_actions()
                # If game.all_actions() is fixed, this works.

                # The notebook logic was:
                # vocab_action_idx = action_idx_to_vocab_idx[legal_action_idx[i]]
                # This assumes legal_action_idx contains INTEGERS that index into all_actions?
                # Let's verify what AlphazeroPlayer returns.
                # AlphazeroPlayer returns 'legal_action_idx' as indices into `game.all_actions()` list.

                # We interpret legal_action_idx[i] as a list of integers.
                current_legal_indices = legal_action_idx[i]
                current_legal_probs = legal_policies[i]

                # We need to turn these game-specific action indices into Vocab indices
                # The vocab was built as [START_OF_GAME] + list(all_actions)
                # So Vocab Index = Game Action Index + 1 (usually, if match is perfect)

                # Using the lookup table we built:
                vocab_indices = action_idx_to_vocab_idx[current_legal_indices]
                fixed_width_policies[i, vocab_indices] = current_legal_probs

            encoded_history = self.action_vocab.encode(action_history)
            tiled_rewards = np.tile(rewards, (length, 1))

            builder.add_trajectory(encoded_history, fixed_width_policies, tiled_rewards)

        save_path = self.data_dir
        # The builder saves to {root}/{split}
        builder.save(str(save_path), f"gen-{gen_id}")

    def train_generation(self, model, gen_id) -> ActionHistoryTransformer:
        """Train model on all data up to gen_id."""

        # Get all training data locations (handling forks)
        dataset_paths = self.get_trajectory_paths(gen_id)

        # Train Config
        train_config = TrainConfig(
            model_name=f"{self.config.experiment_name}",
            model_version=f"gen-{gen_id}",
            device=self.device,
            **self.train_config_dict,
        )

        from rgi.rgizero.train import Trainer
        
        # Build loader using the specs
        train_loader, val_loader = build_trajectory_loader(
            dataset_paths=dataset_paths,
            block_size=self.n_max_context,
            batch_size=train_config.batch_size,
            device=self.device,
            shuffle=True,
        )

        trainer = Trainer(
            model=model, train_config=train_config, train_loader=train_loader, val_loader=val_loader, device=self.device
        )

        trainer.train()

        self.save_model(model, gen_id, {"final_loss": trainer.estimate_loss()})

        return model
