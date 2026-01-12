from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override
from numba import jit

from rgi.rgizero.games.base import Game


@dataclass
class OthelloState:
    board: NDArray[np.int8]  # (height, width)
    current_player: int
    is_terminal: bool = False


GameState = OthelloState
Action = tuple[int, int]
Position = tuple[int, int]
PlayerId = int


class OthelloGame(Game[GameState, Action]):
    """Othello/Reversi game implementation.

    Actions are (row, col) positions where the player can place a piece.
    """

    def __init__(self, board_size: int = 8):
        self.board_size = board_size
        self._all_positions = [(row + 1, col + 1) for row in range(self.board_size) for col in range(self.board_size)]
        self._directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    def _to_grid_coords(self, position: Position) -> tuple[int, int]:
        return (self.board_size - position[0], position[1] - 1)

    def _to_human_coords(self, coordinates: tuple[int, int]) -> Position:
        return (self.board_size - coordinates[0], coordinates[1] + 1)

    def get_piece(self, game_state: OthelloState, position: Position) -> int:
        return game_state.board[self._to_grid_coords(position)].item()  # type: ignore

    @override
    def initial_state(self) -> OthelloState:
        board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        mid = self.board_size // 2
        board[mid - 1, mid - 1] = 2
        board[mid - 1, mid] = 1
        board[mid, mid - 1] = 1
        board[mid, mid] = 2
        return OthelloState(board=board, current_player=1, is_terminal=False)

    @override
    def current_player_id(self, game_state: GameState) -> int:
        return game_state.current_player

    @override
    def num_players(self, game_state: GameState) -> int:
        return 2

    @override
    def legal_actions(self, game_state: OthelloState) -> list[Action]:
        return self._get_legal_moves(game_state, game_state.current_player)

    @override
    def all_actions(self) -> list[Action] | None:
        return self._all_positions

    def next_player(self, player: PlayerId) -> PlayerId:
        return 1 if player == 2 else 2

    def _get_legal_moves(self, game_state: OthelloState, player: PlayerId) -> list[Action]:
        opponent = self.next_player(player)
        # Use numba-optimized version (returns grid coordinates)
        grid_moves = _numba_get_legal_moves(game_state.board, player, opponent, self.board_size)
        # Convert to human coordinates
        return [self._to_human_coords((row, col)) for row, col in grid_moves]

    def _would_flip(self, state: OthelloState, position: Position, player: PlayerId, opponent: PlayerId) -> bool:
        # Use numba-optimized version
        return _numba_would_flip(state.board, position[0], position[1], player, opponent, self.board_size)

    @override
    def next_state(self, game_state: OthelloState, action: Action) -> OthelloState:
        if action not in self.legal_actions(game_state):
            raise ValueError(f"Invalid move: {action} is not a legal action.")

        action = self._to_grid_coords(action)
        player = game_state.current_player
        opponent = self.next_player(player)
        new_board = game_state.board.copy()

        positions_to_flip = self._get_positions_to_flip(game_state, action, player, opponent)

        # Place the new disc
        new_board[action[0], action[1]] = player

        # Flip the opponent's discs
        for pos in positions_to_flip:
            new_board[pos[0], pos[1]] = player

        # Determine next player
        next_player = self.next_player(player)

        # If next player has no legal moves, current player plays again
        next_state = OthelloState(new_board, next_player, is_terminal=False)
        if not self._get_legal_moves(next_state, next_player):
            if self._get_legal_moves(next_state, player):
                next_player = player
            else:
                # No moves for either player; the game ends
                return OthelloState(new_board, player, is_terminal=True)

        return OthelloState(new_board, next_player, is_terminal=False)

    def _get_positions_to_flip(
        self, state: OthelloState, position: Position, player: PlayerId, opponent: PlayerId
    ) -> list[Position]:
        # Use numba-optimized version
        return _numba_get_positions_to_flip(state.board, position[0], position[1], player, opponent, self.board_size)

    @override
    def is_terminal(self, game_state: OthelloState) -> bool:
        return game_state.is_terminal

    @override
    def reward(self, game_state: OthelloState, player_id: PlayerId) -> float | None:
        if not self.is_terminal(game_state):
            return None
        player_count = np.sum(game_state.board == player_id).item()
        opponent_count = np.sum(game_state.board == self.next_player(player_id)).item()
        if player_count > opponent_count:
            return 1.0
        elif player_count < opponent_count:
            return 0.0
        else:
            return 0.5  # Draw

    @override
    def pretty_str(self, game_state: OthelloState) -> str:
        def cell_to_str(cell: int) -> str:
            return "●" if cell == 1 else "○" if cell == 2 else "."

        rows = []
        for row in range(self.board_size):
            row_str = " ".join(cell_to_str(game_state.board[row, col].item()) for col in range(self.board_size))
            rows.append(row_str)

        return "\n".join(rows)

    def parse_board(self, board_str: str, current_player: PlayerId, is_terminal: bool) -> OthelloState:
        """Parses a board string into an OthelloState."""
        board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        rows = board_str.strip().split("\n")
        for r, row in enumerate(rows):
            cells = row.strip().split()
            for c, cell in enumerate(cells):
                if cell == "●":
                    board[r, c] = 1
                elif cell == "○":
                    board[r, c] = 2
        return OthelloState(board=board, current_player=current_player, is_terminal=is_terminal)


# Numba-optimized functions for performance-critical operations
@jit(nopython=True)
def _numba_would_flip(board: NDArray[np.int8], row: int, col: int, player: int, opponent: int, board_size: int) -> bool:
    """Check if placing a piece at (row, col) would flip any opponent pieces."""
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for dr, dc in directions:
        r, c = row + dr, col + dc
        found_opponent = False

        while 0 <= r < board_size and 0 <= c < board_size:
            if board[r, c] == 0:
                break
            elif board[r, c] == opponent:
                found_opponent = True
            elif board[r, c] == player:
                if found_opponent:
                    return True
                else:
                    break
            r += dr
            c += dc

    return False


@jit(nopython=True)
def _numba_get_legal_moves(
    board: NDArray[np.int8], player: int, opponent: int, board_size: int
) -> list[tuple[int, int]]:
    """Get all legal moves for the current player (in grid coordinates)."""
    legal_moves = []

    for row in range(board_size):
        for col in range(board_size):
            if board[row, col] != 0:
                continue  # Skip occupied positions
            if _numba_would_flip(board, row, col, player, opponent, board_size):
                legal_moves.append((row, col))

    return legal_moves


@jit(nopython=True)
def _numba_get_positions_to_flip(
    board: NDArray[np.int8], row: int, col: int, player: int, opponent: int, board_size: int
) -> list[tuple[int, int]]:
    """Get all positions that would be flipped by placing a piece at (row, col)."""
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    positions_to_flip = []

    for dr, dc in directions:
        r, c = row + dr, col + dc
        temp_positions = []

        while 0 <= r < board_size and 0 <= c < board_size:
            if board[r, c] == 0:
                break
            elif board[r, c] == opponent:
                temp_positions.append((r, c))
            elif board[r, c] == player:
                if temp_positions:
                    positions_to_flip.extend(temp_positions)
                break
            r += dr
            c += dc

    return positions_to_flip
