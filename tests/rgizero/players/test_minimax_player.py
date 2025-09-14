"""Tests for MinimaxPlayer."""

from rgi.rgizero.games.count21 import Count21Game, Count21State
from rgi.rgizero.games.connect4 import Connect4Game
from rgi.rgizero.players.minimax_player import MinimaxPlayer


def test_minimax_player_basic():
    """Test basic minimax player functionality."""
    game = Connect4Game()
    player = MinimaxPlayer(game, player_id=1, max_depth=2)
    state = game.initial_state()

    result = player.select_action(state)

    # Check return types
    assert result.action in game.legal_actions(state)
    assert isinstance(result.info, dict)
    assert "eval_score" in result.info


def test_minimax_deterministic():
    """Test that minimax is deterministic."""
    game = Count21Game()
    player = MinimaxPlayer(game, player_id=1, max_depth=3)
    state = game.initial_state()

    result1 = player.select_action(state)
    result2 = player.select_action(state)

    assert result1.action == result2.action


def test_minimax_count21_strategy():
    """Test minimax strategy on Count21."""
    game = Count21Game()
    player = MinimaxPlayer(game, player_id=1, max_depth=5)

    # Test a specific position where optimal play is clear
    # Score 19: Player 1's turn. Best move is 1 to avoid going to 21
    state = Count21State(score=19, current_player=1)

    result = player.select_action(state)

    # Should prefer action 1 to avoid immediate loss
    assert result.action == 1


def test_minimax_initialization():
    """Test minimax player initialization."""
    game = Count21Game()
    player = MinimaxPlayer(game, player_id=42, max_depth=5)

    assert player.player_id == 42
    assert player.max_depth == 5
    assert player.game is game


def test_minimax_depth_effect():
    """Test that different depths can affect decisions."""
    game = Count21Game()

    player_shallow = MinimaxPlayer(game, player_id=1, max_depth=1)
    player_deep = MinimaxPlayer(game, player_id=1, max_depth=5)

    state = game.initial_state()

    result1 = player_shallow.select_action(state)
    result2 = player_deep.select_action(state)

    # Both should return valid actions
    assert result1.action in game.legal_actions(state)
    assert result2.action in game.legal_actions(state)

    # Different depths might choose different actions
    # (not guaranteed, but likely on a complex game)
