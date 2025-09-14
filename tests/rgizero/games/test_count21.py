import pytest

from rgi.rgizero.games.count21 import Count21Game, Count21State

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive


@pytest.fixture
def game() -> Count21Game:
    return Count21Game()


def test_initial_state(game: Count21Game) -> None:
    assert game.initial_state() == Count21State(0, 1)


def test_current_player_id(game: Count21Game) -> None:
    assert game.current_player_id(Count21State(0, 1)) == 1
    assert game.current_player_id(Count21State(0, 2)) == 2


def test_legal_actions(game: Count21Game) -> None:
    assert game.legal_actions(Count21State(0, 1)) == (1, 2, 3)


def test_next_state(game: Count21Game) -> None:
    assert game.next_state(Count21State(0, 1), 3) == Count21State(3, 2)
    assert game.next_state(Count21State(3, 2), 3) == Count21State(6, 1)
    assert game.next_state(Count21State(6, 1), 1) == Count21State(7, 2)


def test_is_terminal(game: Count21Game) -> None:
    assert not game.is_terminal(Count21State(0, 1))
    assert not game.is_terminal(Count21State(20, 1))
    assert game.is_terminal(Count21State(21, 1))


def test_reward(game: Count21Game) -> None:
    # non-terminal states
    assert game.reward(Count21State(20, 1), 1) == 0.0
    assert game.reward(Count21State(20, 1), 2) == 0.0

    # terminal state
    assert game.reward(Count21State(21, 1), 1) == 1.0
    assert game.reward(Count21State(21, 1), 2) == -1.0


def test_pretty_str(game: Count21Game) -> None:
    assert game.pretty_str(Count21State(9, 1)) == "Score: 9, Player: 1"
