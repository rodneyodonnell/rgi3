"""Tests for RandomPlayer."""

from rgi.rgizero.games.count21 import Count21Game
from rgi.rgizero.players.random_player import RandomPlayer


def test_random_player_basic():
    """Test basic random player functionality."""
    game = Count21Game()
    player = RandomPlayer(game, seed=42)
    state = game.initial_state()

    result = player.select_action(state)

    # Check return types
    assert result.action in game.legal_actions(state)
    assert isinstance(result.info, dict)
    assert "legal_policy" in result.info
    assert "legal_actions" in result.info


def test_random_player_deterministic_with_seed():
    """Test that random player is deterministic with same seed."""
    game = Count21Game()

    player1 = RandomPlayer(game, seed=42)
    player2 = RandomPlayer(game, seed=42)

    state = game.initial_state()

    result1 = player1.select_action(state)
    result2 = player2.select_action(state)

    assert result1.action == result2.action


def test_random_player_different_seeds():
    """Test that different seeds can produce different results."""
    game = Count21Game()
    state = game.initial_state()

    # Run multiple times to increase chance of different results
    actions = []
    for seed in range(10):
        player = RandomPlayer(game, seed=seed)
        result = player.select_action(state)
        actions.append(result.action)

    # Should have some variation (not all the same)
    unique_actions = set(actions)
    assert len(unique_actions) > 1
