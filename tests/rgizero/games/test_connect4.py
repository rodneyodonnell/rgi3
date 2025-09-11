import textwrap
from typing import Literal

import numpy as np
import pytest

from rgi.rgizero.games.connect4 import Connect4Game

TPlayerId = Literal[1, 2]

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive


@pytest.fixture
def game() -> Connect4Game:
    return Connect4Game()


def test_initial_state(game: Connect4Game) -> None:
    state = game.initial_state()
    assert state.current_player == 1
    assert np.all(state.board == 0)


def test_legal_actions(game: Connect4Game) -> None:
    state = game.initial_state()
    assert game.legal_actions(state) == tuple(range(1, 8))

    # Fill up a column
    for _ in range(6):
        state = game.next_state(state, action=1)
    assert game.legal_actions(state) == tuple(range(2, 8))


def test_next_state(game: Connect4Game) -> None:
    state = game.initial_state()
    next_state = game.next_state(state, action=3)
    assert next_state.current_player == 2
    assert next_state.board[5, 2] == 1  # Remember, it's zero-indexed now


def test_is_terminal(game: Connect4Game) -> None:
    state = game.initial_state()
    assert not game.is_terminal(state)

    # Create a winning state
    for i in range(1, 4):
        state = game.next_state(state, action=i)
        assert not game.is_terminal(state)
        state = game.next_state(state, action=i)
        assert not game.is_terminal(state)
    state = game.next_state(state, action=4)
    print(game.pretty_str(state))
    assert game.is_terminal(state)


def test_reward(game: Connect4Game) -> None:
    state = game.initial_state()
    assert game.reward(state, 1) == 0
    assert game.reward(state, 2) == 0

    # Create a winning state for player 1
    for i in range(4):
        state = game.next_state(state, action=i + 1)
        if i < 3:
            state = game.next_state(state, action=i + 1)
    assert game.reward(state, 1) == 1
    assert game.reward(state, 2) == -1


def test_vertical_win(game: Connect4Game) -> None:
    state = game.initial_state()
    for _ in range(3):
        state = game.next_state(state, action=1)
        state = game.next_state(state, action=2)
    state = game.next_state(state, action=1)
    assert game.is_terminal(state)
    assert game.reward(state, 1) == 1


def test_horizontal_win(game: Connect4Game) -> None:
    state = game.initial_state()
    for i in range(4):
        state = game.next_state(state, action=i + 1)
        if i < 3:
            state = game.next_state(state, action=1)
    assert game.is_terminal(state)
    assert game.reward(state, 1) == 1


def test_diagonal_win(game: Connect4Game) -> None:
    state = game.initial_state()
    moves = [0, 1, 1, 2, 2, 3, 2, 3, 3, 0, 3]
    for move in moves:
        state = game.next_state(state, move + 1)
    assert game.is_terminal(state)
    assert game.reward(state, 1) == 1


def test_invalid_move(game: Connect4Game) -> None:
    state = game.initial_state()
    with pytest.raises(ValueError):
        game.next_state(state, 8)


def test_custom_board_size() -> None:
    game = Connect4Game(width=8, height=7, connect_length=5)
    state = game.initial_state()
    assert len(game.legal_actions(state)) == 8
    for _ in range(7):
        state = game.next_state(state, 1)
    assert len(game.legal_actions(state)) == 7


def test_pretty_str(game: Connect4Game) -> None:
    state = game.initial_state()
    for action in [1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 7, 7]:
        state = game.next_state(state, action)

    assert (
        game.pretty_str(state).strip()
        == textwrap.dedent(
            """
            | | | | | | | |
            | | | | | | | |
            | | | | | | | |
            | | |○| | | |●|
            | |●|●|○| | |○|
            |●|○|○|●|●|○|●|
            +-+-+-+-+-+-+-+
            """
        ).strip()
    )


@pytest.mark.parametrize("verbose", [True, False])
def test_draw(game: Connect4Game, verbose: bool) -> None:
    state = game.initial_state()
    # fmt: off
    moves = [
        1, 2, 3, 4, 5, 6,
        1, 2, 3, 4, 5, 6,
        1, 2, 3, 4, 5, 6, 7,
        1, 2, 3, 4, 5, 6,
        1, 2, 3, 4, 5, 6,
        1, 2, 3, 4, 5, 6,
        7, 7, 7, 7,
    ]
    # fmt: on
    for i, move in enumerate(moves):
        state = game.next_state(state, move)
        assert not game.is_terminal(state)
        assert game.reward(state, 1) == 0
        assert game.reward(state, 2) == 0
        if verbose:
            print(f"Debug - Move {i + 1}: {move}")
            print(game.pretty_str(state))
            print(f"Is terminal: {game.is_terminal(state)}\n")

    state = game.next_state(state, 7)
    assert game.is_terminal(state)
    assert game.reward(state, 1) == 0
    assert game.reward(state, 2) == 0


@pytest.mark.parametrize(
    "board_str, expected_player",
    [
        (
            textwrap.dedent(
                """
    | | | | | | | |
    | | | | | | | |
    | | | | | | | |
    | | | | | | | |
    | | | | | | | |
    | | | | | | | |
    +-+-+-+-+-+-+-+
    """
            ),
            2,
        ),
        (
            textwrap.dedent(
                """
    | | | | | | | |
    | | | | | | | |
    | | | | | | | |
    | | | | | | |●|
    |○| | | | | |●|
    |○|○| | | | |●|
    +-+-+-+-+-+-+-+
    """
            ),
            2,
        ),
        (
            textwrap.dedent(
                """
    |○| | | | | | |
    |○| | |●| | | |
    |●| | |○| | | |
    |○|○| |●| | | |
    |○|○|●|●| |●| |
    |○|●|○|●|●|●|○|
    +-+-+-+-+-+-+-+
    """
            ),
            2,
        ),
    ],
)
def test_connect4_parse_and_pretty_print(game: Connect4Game, board_str: str, expected_player: TPlayerId) -> None:
    state = game.parse_board(board_str, current_player=expected_player)
    pretty_printed = game.pretty_str(state)
    assert pretty_printed.strip() == board_str.strip()


def test_middle_of_row_win() -> None:
    game = Connect4Game()

    board_str = textwrap.dedent(
        """
        | | | | | | | |
        | | | | | | | |
        | | | | | | | |
        |○| | | | | |○|
        |○| | | | | |○|
        |○|●|○|●|●| |●|
        +-+-+-+-+-+-+-+
        """
    )
    state = game.parse_board(board_str, current_player=1)
    assert state.winner == 0

    new_state = game.next_state(state, 6)
    assert new_state.winner == 1, f"Expected Player 1 to win, but got {state.winner}"


def test_state_immutability(game: Connect4Game) -> None:
    initial_state = game.initial_state()
    assert np.all(initial_state.board == 0)

    updated_state = game.next_state(initial_state, 3)
    assert np.all(initial_state.board == 0)
    assert initial_state.current_player == 1
    assert not np.all(updated_state.board == 0)
    assert updated_state.current_player == 2
