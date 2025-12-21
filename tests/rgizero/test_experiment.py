import pytest
import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import torch

from rgi.rgizero.experiment import ExperimentConfig, ExperimentRunner


@pytest.fixture
def base_dir(tmp_path):
    return tmp_path / "experiments"


@pytest.fixture
def config():
    return ExperimentConfig(
        experiment_name="test_exp",
        game_name="connect4",
        num_generations=5,
        num_games_per_gen=10,
        num_simulations=5,
        model_size="tiny",
        train_batch_size=4,
    )


def test_config_serialization(config):
    json_data = config.to_json()
    config2 = ExperimentConfig.from_json(json_data)
    assert config == config2


def test_runner_init_directories(base_dir, config):
    runner = ExperimentRunner(config, base_dir)

    assert (base_dir / "test_exp" / "data").exists()
    assert (base_dir / "test_exp" / "models").exists()
    assert (base_dir / "test_exp" / "config.json").exists()


def test_path_resolution_simple(base_dir, config):
    runner = ExperimentRunner(config, base_dir)

    # Simple check - should point to local
    p = runner.get_trajectory_path(1)
    assert p == base_dir / "test_exp" / "data" / "gen-1"


def test_path_resolution_forking(base_dir, config):
    # Setup Parent
    parent_name = "parent_exp"
    parent_dir = base_dir / parent_name / "data"
    os.makedirs(parent_dir)

    # Create fake data in parent
    (parent_dir / "gen-1").touch()
    (parent_dir / "gen-2").touch()

    # Setup Child config
    config.parent_experiment_name = parent_name
    config.parent_generation_cap = 1  # Only inherit gen 1

    runner = ExperimentRunner(config, base_dir)

    # Gen 1 should be inherited (exists in parent, <= cap)
    p1 = runner.get_trajectory_path(1)
    assert p1 == parent_dir / "gen-1"

    # Gen 2 should NOT be inherited (exists in parent, but > cap)
    p2 = runner.get_trajectory_path(2)
    assert p2 == base_dir / "test_exp" / "data" / "gen-2"

    # Gen 3 should NOT be inherited (does not exist in parent)
    p3 = runner.get_trajectory_path(3)
    assert p3 == base_dir / "test_exp" / "data" / "gen-3"


def test_path_resolution_local_override(base_dir, config):
    # Setup Parent
    parent_name = "parent_exp"
    parent_dir = base_dir / parent_name / "data"
    os.makedirs(parent_dir)
    (parent_dir / "gen-1").touch()

    # Setup Local override
    local_dir = base_dir / "test_exp" / "data"
    os.makedirs(local_dir)
    (local_dir / "gen-1").touch()

    config.parent_experiment_name = parent_name
    config.parent_generation_cap = 5

    runner = ExperimentRunner(config, base_dir)

    # Should prefer local if it exists
    p1 = runner.get_trajectory_path(1)
    assert p1 == local_dir / "gen-1"


import os
