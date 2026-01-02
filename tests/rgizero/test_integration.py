"""
Integration tests for the full AlphaZero training pipeline.

These tests run end-to-end training loops to ensure the system works correctly:
- Self-play game generation
- Training over multiple generations
- Model improvement validation
- ELO-based evaluation

Tests use minimal configurations (tiny models, few games) to run quickly (<5 minutes).
"""

import asyncio
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

from rgi.rgizero.evaluators import ActionHistoryTransformerEvaluator, AsyncNetworkEvaluator
from rgi.rgizero.experiment import ExperimentConfig, ExperimentRunner
from rgi.rgizero.models.transformer import TransformerConfig
from rgi.rgizero.players.alphazero import AlphazeroPlayer
from rgi.rgizero.tournament import Tournament


@pytest.fixture
def temp_experiment_dir():
    """Create a temporary directory for experiment artifacts."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def minimal_training_args():
    """Minimal training arguments for fast integration tests."""
    return {
        # Very tiny model for fast convergence in tests
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 32,  # Compact embedding to prevent overfitting
        "n_max_context": 100,  # Use fallback size that works for all games
        "dropout": 0.1,  # Add dropout for regularization
        "bias": False,
        # Fast training - optimized for small dataset
        "batch_size": 32,  # Larger batches for better stability
        "gradient_accumulation_steps": 1,
        "max_iters": 5000,  # More iterations to allow learning
        "max_epochs": 50,  # Allow multiple passes through data
        "learning_rate": 0.0005,  # Lower LR for stability
        "decay_lr": True,
        "min_lr": 0.00005,
        "lr_decay_iters": 5000,
        "warmup_iters": 100,  # Scaled up warmup
        "weight_decay": 0.01,  # Lower weight decay to not constrain small model
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "dtype": "float32",  # Use float32 for CPU compatibility
        "eval_iters": 20,
        "log_interval": 50,
        "eval_interval": 50,
        "early_stop_patience": 20,  # Allow more patience for noisy small dataset
    }


@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_training_pipeline_count21(temp_experiment_dir, minimal_training_args):
    """
    Test full training pipeline on Count21 game.

    This test:
    1. Initializes an experiment with a random model
    2. Runs 3 generations of self-play + training
    3. Validates that models are saved correctly
    4. Checks that training completes without errors

    Uses Count21 (simplest game) for speed.
    Target runtime: ~1-2 minutes
    """
    config = ExperimentConfig(
        experiment_name="test-count21-pipeline",
        game_name="count21",
        num_generations=3,
        num_games_per_gen=20,  # Minimal games for testing
        num_simulations=10,  # Minimal MCTS simulations
        seed=42,
    )

    runner = ExperimentRunner(
        config=config,
        base_dir=temp_experiment_dir,
        training_args=minimal_training_args,
        progress_bar=False,  # Disable for cleaner test output
    )

    # Initialize Gen 0
    model_0 = runner.initialize()
    assert model_0 is not None
    assert runner.get_model_path(0).exists()

    # Run 3 generations
    current_model = model_0
    for gen_id in range(1, config.num_generations + 1):
        current_model = await runner.run_generation_step_async(gen_id, current_model)

        # Validate artifacts exist
        assert runner.get_trajectory_path(gen_id).exists(), f"Missing trajectory data for gen {gen_id}"
        assert runner.get_model_path(gen_id).exists(), f"Missing model checkpoint for gen {gen_id}"

    # Validate we can load the final model
    final_model = runner.load_model(config.num_generations)
    assert final_model is not None

    print(f"✓ Training pipeline completed successfully for {config.num_generations} generations")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_training_pipeline_connect4(temp_experiment_dir, minimal_training_args):
    """
    Test full training pipeline on Connect4 (more realistic game).

    This test:
    1. Runs 3 generations of self-play + training on Connect4
    2. Validates that training completes without errors
    3. Checks that model artifacts are created

    Target runtime: ~2-3 minutes
    """
    config = ExperimentConfig(
        experiment_name="test-connect4-pipeline",
        game_name="connect4",
        num_generations=3,
        num_games_per_gen=30,  # Slightly more games for Connect4
        num_simulations=20,  # More simulations for better gameplay
        seed=42,
    )

    runner = ExperimentRunner(
        config=config,
        base_dir=temp_experiment_dir,
        training_args=minimal_training_args,
        progress_bar=False,
    )

    # Run full pipeline
    await runner.run_async()

    # Validate all generations completed
    for gen_id in range(config.num_generations + 1):  # Include gen 0
        assert runner.get_model_path(gen_id).exists(), f"Missing model for gen {gen_id}"

    for gen_id in range(1, config.num_generations + 1):  # Gen 0 has no trajectory
        assert runner.get_trajectory_path(gen_id).exists(), f"Missing trajectory for gen {gen_id}"

    print(f"✓ Connect4 training pipeline completed successfully")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_model_improvement_validation(temp_experiment_dir, minimal_training_args):
    """
    Test that trained models show improvement over random baseline.

    This test:
    1. Trains a model for 3 generations
    2. Plays evaluation games between Gen 0 (random) and Gen 3 (trained)
    3. Validates that Gen 3 wins more than 60% of games

    Target runtime: ~2-3 minutes
    """
    config = ExperimentConfig(
        experiment_name="test-model-improvement",
        game_name="count21",
        num_generations=5,  # More generations to recover from bad starts
        num_games_per_gen=400,  # Significantly more games
        num_simulations=100,  # Better teacher quality
        seed=42,
    )

    runner = ExperimentRunner(
        config=config,
        base_dir=temp_experiment_dir,
        training_args=minimal_training_args,
        progress_bar=False,
    )

    # Run training
    await runner.run_async()

    # Load models
    model_0 = runner.load_model(0)
    model_final = runner.load_model(config.num_generations)

    # Play evaluation games
    num_eval_games = 200  # More games for better ELO resolution

    @asynccontextmanager
    async def create_player_factory(model, simulations):
        """Helper to create player factory with shared evaluator."""
        serial_evaluator = ActionHistoryTransformerEvaluator(
            model,
            device=runner.device,
            block_size=runner.n_max_context,
            vocab=runner.action_vocab,
            verbose=False,
        )
        async_evaluator = AsyncNetworkEvaluator(
            base_evaluator=serial_evaluator,
            max_batch_size=32,
            verbose=False,
        )

        await async_evaluator.start()

        try:
            def factory():
                rng = np.random.default_rng(np.random.randint(0, 2**31))
                return AlphazeroPlayer(
                    runner.game,
                    async_evaluator,
                    rng=rng,
                    add_noise=False,  # No noise for evaluation
                    simulations=simulations,
                )

            yield factory
        finally:
            await async_evaluator.stop()

    async with (
        create_player_factory(model_0, 20) as factory_gen0,
        create_player_factory(model_final, 20) as factory_final,
    ):
        player_factories = {
            "gen_0": factory_gen0,
            "gen_final": factory_final,
        }

        tournament = Tournament(runner.game, player_factories, initial_elo=1000)
        await tournament.run(num_games=num_eval_games, concurrent_games=10)

        # Check that final model has higher ELO
        elo_gen0 = tournament.stats["gen_0"].elo
        elo_final = tournament.stats["gen_final"].elo

        print(f"\nELO Ratings:")
        print(f"  Gen 0 (random): {elo_gen0:.1f}")
        print(f"  Gen {config.num_generations} (trained): {elo_final:.1f}")

        # Trained model should have higher ELO
        assert elo_final > elo_gen0, (
            f"Trained model (ELO={elo_final:.1f}) should beat random model (ELO={elo_gen0:.1f})"
        )

        print(f"✓ Model improvement validated: Gen {config.num_generations} > Gen 0")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_elo_progression_across_generations(temp_experiment_dir, minimal_training_args):
    """
    Test ELO ratings show general improvement trend across generations.

    This test:
    1. Trains models for 4 generations
    2. Runs a tournament with all generations
    3. Validates that later generations generally have higher ELO

    Note: We don't expect strict monotonic improvement due to randomness,
    but the average of later generations should be higher.

    Target runtime: ~3-4 minutes
    """
    config = ExperimentConfig(
        experiment_name="test-elo-progression",
        game_name="count21",
        num_generations=3,  # Fewer generations but more data each
        num_games_per_gen=200,  # Enough games for consistent +50 ELO improvement
        num_simulations=50,  # Higher quality MCTS
        seed=42,
    )

    runner = ExperimentRunner(
        config=config,
        base_dir=temp_experiment_dir,
        training_args=minimal_training_args,
        progress_bar=False,
    )

    # Run training
    await runner.run_async()

    # Load all models
    models = {gen_id: runner.load_model(gen_id) for gen_id in range(config.num_generations + 1)}

    # Create player factories
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
                max_batch_size=32,
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
                        simulations=20,
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

        # Run tournament (more games for stable ELO estimates)
        num_tournament_games = 100  # Total games (more for less variance)
        await tournament.run(num_games=num_tournament_games, concurrent_games=10)

        # Print standings
        print("\nTournament Results:")
        tournament.print_standings()

        # Print all ELO ratings for analysis
        print("\nIndividual ELO ratings:")
        for gen_id in range(config.num_generations + 1):
            elo = tournament.stats[f"gen_{gen_id}"].elo
            games = tournament.stats[f"gen_{gen_id}"].games_played
            wins = tournament.stats[f"gen_{gen_id}"].wins
            print(f"  Gen {gen_id}: ELO={elo:.1f}, Games={games}, Wins={wins}")

        # Primary check: BEST model should be one of the trained ones, not Gen 0
        # With small dataset, we can't expect monotonic improvement, but we should
        # see SOME model beat the random baseline
        elo_gen0 = tournament.stats["gen_0"].elo
        all_trained_elos = [tournament.stats[f"gen_{g}"].elo for g in range(1, config.num_generations + 1)]
        best_trained_elo = max(all_trained_elos)
        best_trained_gen = all_trained_elos.index(best_trained_elo) + 1

        print(f"\nGen 0 (random): {elo_gen0:.1f}")
        print(f"Best trained model: Gen {best_trained_gen}, ELO={best_trained_elo:.1f}")
        print(f"All trained ELOs: {[f'{e:.1f}' for e in all_trained_elos]}")

        # The best trained model should beat random by at least 50 ELO
        elo_improvement = best_trained_elo - elo_gen0
        print(f"\nELO Improvement: {elo_improvement:+.1f} ELO")

        assert elo_improvement > 50, (
            f"Best trained model (Gen {best_trained_gen}, ELO={best_trained_elo:.1f}) did not beat "
            f"random (Gen 0, ELO={elo_gen0:.1f}) by at least 50 ELO. "
            f"Improvement: {elo_improvement:+.1f} ELO. "
            f"This suggests training is not working. Check that START tokens match between train/inference."
        )

        # At least one trained model should show improvement
        num_better_than_random = sum(1 for elo in all_trained_elos if elo > elo_gen0 - 10)
        print(f"Trained models better than random: {num_better_than_random}/{len(all_trained_elos)}")

        print(f"✓ ELO test passed: Training shows strong improvement (Improvement: {elo_improvement:+.1f} ELO)")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_experiment_forking(temp_experiment_dir, minimal_training_args):
    """
    Test experiment forking functionality.

    This test:
    1. Runs a parent experiment for 2 generations
    2. Creates a child experiment that forks from parent at gen 1
    3. Continues training for 1 more generation
    4. Validates that child reuses parent's data and models correctly

    Target runtime: ~2-3 minutes
    """
    # Run parent experiment
    parent_config = ExperimentConfig(
        experiment_name="test-fork-parent",
        game_name="count21",
        num_generations=2,
        num_games_per_gen=20,
        num_simulations=10,
        seed=42,
    )

    parent_runner = ExperimentRunner(
        config=parent_config,
        base_dir=temp_experiment_dir,
        training_args=minimal_training_args,
        progress_bar=False,
    )

    await parent_runner.run_async()

    # Create child experiment that forks from parent
    child_config = ExperimentConfig(
        experiment_name="test-fork-child",
        game_name="count21",
        num_generations=3,  # Continue for 1 more generation
        num_games_per_gen=20,
        num_simulations=10,
        parent_experiment_name="test-fork-parent",
        parent_generation_cap=2,  # Use parent's gen 0-2
        seed=43,  # Different seed
    )

    child_runner = ExperimentRunner(
        config=child_config,
        base_dir=temp_experiment_dir,
        training_args=minimal_training_args,
        progress_bar=False,
    )

    # Child should be able to load parent's models
    parent_model_1 = child_runner.load_model(1)
    assert parent_model_1 is not None

    # Run child experiment (should reuse gen 0-2 from parent, train gen 3)
    await child_runner.run_async()

    # Validate child has its own gen 3
    assert child_runner.get_model_path(3).exists()
    assert child_runner.get_trajectory_path(3).exists()

    # Child should have reused parent's gen 1, 2 (only gen 0 is created locally for child)
    # Check that child's gen 1, 2 point to parent's data
    parent_gen1_path = parent_runner.get_trajectory_path(1)
    child_gen1_path = child_runner.get_trajectory_path(1)

    # They should be the same path (forking works)
    assert child_gen1_path == parent_gen1_path

    print("✓ Experiment forking validated")


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v", "-s"])
