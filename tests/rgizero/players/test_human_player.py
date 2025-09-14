"""Tests for HumanPlayer."""

from unittest.mock import patch

import pytest

from rgi.rgizero.games.count21 import Count21Game
from rgi.rgizero.players.human_player import HumanPlayer


@pytest.mark.parametrize(
    "user_input, expected_action",
    [
        ("1", 1),
        ("2", 2),
        ("3", 3),
        ("i:1", 1),
        ("i:2", 2),
        ("i:3", 3),
    ],
)
def test_select_action(user_input: str, expected_action: int):
    """Test human player action selection with various input formats."""
    game = Count21Game()
    player = HumanPlayer(game)
    state = game.initial_state()

    with patch("builtins.input", return_value=user_input):
        result = player.select_action(state)
        assert result.action == expected_action


def test_invalid_input():
    """Test human player handles invalid input gracefully."""
    game = Count21Game()
    player = HumanPlayer(game)
    state = game.initial_state()

    with patch("builtins.input", side_effect=["invalid", "1"]):
        result = player.select_action(state)
        assert result.action == 1


def test_human_player_initialization():
    """Test human player initialization."""
    game = Count21Game()
    player = HumanPlayer(game)
    assert player.game is game
