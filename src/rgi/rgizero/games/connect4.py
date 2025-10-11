from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override
from numba import jit

from rgi.rgizero.games.base import Game


@dataclass
class Connect4State:
    board: NDArray[np.int8]  # (height, width)
    current_player: int
    is_terminal: bool
    winner: int | None  # The winner, if the game has ended. 0 for draw. None if game is not terminal.


PlayerId = int
GameState = Connect4State
Action = int


class Connect4Game(Game[GameState, Action]):
    """Connect 4 game implementation.

    Actions are column numbers (1-7) where the player can drop a piece.
    """

    def __init__(self, width: int = 7, height: int = 6, connect_length: int = 4):
        self.width = width
        self.height = height
        self.connect_length = connect_length
        self._all_column_ids = tuple(range(1, width + 1))
        self._all_row_ids = tuple(range(1, height + 1))

    @override
    def initial_state(self) -> GameState:
        return GameState(
            board=np.zeros([self.height, self.width], dtype=np.int8),
            current_player=1,
            is_terminal=False,
            winner=None,
        )

    @override
    def current_player_id(self, game_state: GameState) -> int:
        return game_state.current_player

    @override
    def num_players(self, game_state: GameState) -> int:
        return 2

    @override
    def legal_actions(self, game_state: GameState) -> Sequence[Action]:
        return tuple(col + 1 for col in range(self.width) if game_state.board[0, col] == 0)

    @override
    def all_actions(self) -> Sequence[Action]:
        return self._all_column_ids

    @override
    def next_state(self, game_state: GameState, action: Action) -> GameState:
        """Find the lowest empty row in the selected column and return the updated game state."""
        new_board, next_player, is_terminal, winner = _numba_next_state(
            game_state.board, action, game_state.current_player, self.height, self.width, self.connect_length
        )

        return GameState(board=new_board, current_player=next_player, is_terminal=is_terminal, winner=winner)

    def _calculate_winner(self, board: NDArray[np.int8], col: int, row: int, player: PlayerId) -> PlayerId | None:
        """Check if the last move made at (row, col) by 'player' wins the game."""
        return _numba_calculate_winner(board, col, row, player, self.height, self.width, self.connect_length)

    def _calculate_is_board_full(self, board: NDArray[np.int8]) -> bool:
        return _numba_calculate_is_board_full(board)

    @override
    def is_terminal(self, game_state: GameState) -> bool:
        return game_state.is_terminal

    @override
    def reward(self, game_state: GameState, player_id: PlayerId) -> float | None:
        # No reward for non-terminal states
        if not self.is_terminal(game_state):
            return None

        # Draw
        if game_state.winner is None:
            return 0.5

        if game_state.winner == player_id:
            return 1.0
        return 0.0

    @override
    def pretty_str(self, game_state: GameState) -> str:
        symbols = [" ", "●", "○"]
        return (
            "\n".join("|" + "|".join(symbols[int(cell)] for cell in row) + "|" for row in game_state.board)
            + "\n+"
            + "-+" * self.width
        )

    def parse_board(self, board_str: str, current_player: PlayerId) -> GameState:
        """Parses the output of pretty_str into a GameState."""
        rows = board_str.strip().split("\n")[:-1]  # Skip the bottom border row
        board = np.zeros((self.height, self.width), dtype=np.int8)
        for r, row in enumerate(rows):
            row_cells = row.strip().split("|")[1:-1]  # Extract cells between borders
            for c, cell in enumerate(row_cells):
                if cell == "●":
                    board[r, c] = 1  # Player 1
                elif cell == "○":
                    board[r, c] = 2  # Player 2
        # TODO: Handle terminal & final states properly...
        return GameState(board=board, current_player=current_player, is_terminal=False, winner=None)


@jit(nopython=True)
def _numba_next_state(
    board: NDArray[np.int8],
    action: int,
    current_player: PlayerId,
    height: int,
    width: int,
    connect_length: int,
) -> tuple[NDArray[np.int8], PlayerId, bool, PlayerId | None]:
    """Find the lowest empty row in the selected column and return the updated game state."""
    column = action - 1
    if not (0 <= column < width and board[0, column] == 0):
        raise ValueError(f"Invalid move: Column {action} is full or out of bounds.")

    row = np.nonzero(board[:, column] == 0)[0][-1]

    new_board = board.copy()
    new_board[row, column] = current_player

    winner = _numba_calculate_winner(new_board, column, row, current_player, height, width, connect_length)
    is_terminal = (winner is not None) or _numba_calculate_is_board_full(new_board)
    next_player: PlayerId = 2 if current_player == 1 else 1

    return new_board, next_player, is_terminal, winner


@jit(nopython=True)
def _numba_calculate_winner(
    board: NDArray[np.int8], col: int, row: int, player: PlayerId, height: int, width: int, connect_length: int
) -> PlayerId | None:
    """Check if the last move made at (row, col) by 'player' wins the game."""
    directions = [
        (0, 1),  # Horizontal
        (1, 0),  # Vertical
        (1, 1),  # Diagonal /
        (1, -1),  # Diagonal \
    ]

    for dr, dc in directions:
        count = 1
        for factor in [-1, 1]:
            r, c = row + dr * factor, col + dc * factor
            while 0 <= r < height and 0 <= c < width and board[r, c] == player:
                count += 1
                r, c = r + dr * factor, c + dc * factor
                if count >= connect_length:
                    return player

    return None


@jit(nopython=True)
def _numba_calculate_is_board_full(board: NDArray[np.int8]) -> bool:
    return np.all(board != 0).item()
