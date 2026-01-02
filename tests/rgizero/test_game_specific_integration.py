"""
Integration tests for specific games (Othello, Count21, Connect4).

These tests validate the full training pipeline works for each game.
"""

import tempfile
import shutil
from pathlib import Path

import pytest

from rgi.rgizero.experiment import ExperimentConfig, ExperimentRunner


@pytest.fixture
def temp_experiment_dir():
    """Create a temporary directory for experiment artifacts."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def minimal_training_args():
    """Minimal training arguments for fast integration tests."""
    return {
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 8,
        "n_max_context": 100,
        "dropout": 0.0,
        "bias": False,
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "max_iters": 300,
        "max_epochs": 10,
        "learning_rate": 0.005,
        "decay_lr": True,
        "min_lr": 0.0005,
        "lr_decay_iters": 300,
        "warmup_iters": 10,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "dtype": "float32",
        "eval_iters": 20,
        "log_interval": 50,
        "eval_interval": 50,
        "early_stop_patience": 5,
    }


@pytest.mark.asyncio
@pytest.mark.integration
async def test_count21_full_pipeline(temp_experiment_dir, minimal_training_args):
    """Test Count21 game trains successfully."""
    config = ExperimentConfig(
        experiment_name="test-count21",
        game_name="count21",
        num_generations=2,
        num_games_per_gen=40,
        num_simulations=20,
        seed=42,
    )

    runner = ExperimentRunner(
        config=config,
        base_dir=temp_experiment_dir,
        training_args=minimal_training_args,
        progress_bar=False,
    )

    await runner.run_async()

    # Validate all artifacts exist
    for gen_id in range(config.num_generations + 1):
        assert runner.get_model_path(gen_id).exists(), f"Missing model for gen {gen_id}"

    print(f"✓ Count21 training completed successfully for {config.num_generations} generations")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_connect4_full_pipeline(temp_experiment_dir, minimal_training_args):
    """Test Connect4 game trains successfully."""
    config = ExperimentConfig(
        experiment_name="test-connect4",
        game_name="connect4",
        num_generations=2,
        num_games_per_gen=40,
        num_simulations=20,
        seed=42,
    )

    runner = ExperimentRunner(
        config=config,
        base_dir=temp_experiment_dir,
        training_args=minimal_training_args,
        progress_bar=False,
    )

    await runner.run_async()

    # Validate all artifacts exist
    for gen_id in range(config.num_generations + 1):
        assert runner.get_model_path(gen_id).exists(), f"Missing model for gen {gen_id}"

    print(f"✓ Connect4 training completed successfully for {config.num_generations} generations")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_othello_full_pipeline(temp_experiment_dir, minimal_training_args):
    """Test Othello game trains successfully."""
    config = ExperimentConfig(
        experiment_name="test-othello",
        game_name="othello",
        num_generations=2,
        num_games_per_gen=40,
        num_simulations=20,
        seed=42,
    )

    runner = ExperimentRunner(
        config=config,
        base_dir=temp_experiment_dir,
        training_args=minimal_training_args,
        progress_bar=False,
    )

    await runner.run_async()

    # Validate all artifacts exist
    for gen_id in range(config.num_generations + 1):
        assert runner.get_model_path(gen_id).exists(), f"Missing model for gen {gen_id}"

    print(f"✓ Othello training completed successfully for {config.num_generations} generations")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
