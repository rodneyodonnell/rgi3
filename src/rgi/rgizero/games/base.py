from abc import ABC, abstractmethod
from typing import Any, Generic, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray

TGame = TypeVar("TGame", bound="Game[Any, Any]")  # pylint: disable=invalid-name
TGameState = TypeVar("TGameState")  # pylint: disable=invalid-name
TPlayerState = TypeVar("TPlayerState")  # pylint: disable=invalid-name
TAction = TypeVar("TAction")  # pylint: disable=invalid-name
TPlayerData = TypeVar("TPlayerData")  # pylint: disable=invalid-name
TPlayerId = int  # pylint: disable=invalid-name


class Game(ABC, Generic[TGameState, TAction]):
    @abstractmethod
    def initial_state(self) -> TGameState:
        """Create an initial game state."""

    @abstractmethod
    def current_player_id(self, game_state: TGameState) -> TPlayerId:
        """Return the ID of the current player. Sequential starting at 1."""

    @abstractmethod
    def num_players(self, game_state: TGameState) -> int:
        """Number of players in the game."""

    def player_ids(self, game_state: TGameState) -> Sequence[TPlayerId]:
        """Return a sequence of all player IDs in the game."""
        return range(1, self.num_players(game_state) + 1)

    @abstractmethod
    def legal_actions(self, game_state: TGameState) -> Sequence[TAction]:
        """Return a sequence of all legal actions for the game state."""

    @abstractmethod
    def all_actions(self) -> Sequence[TAction]:
        """Return a sequence of all possible actions in the game."""

    @abstractmethod
    def next_state(self, game_state: TGameState, action: TAction) -> TGameState:
        """Return a new immutable game state. Must not modify the input state."""

    @abstractmethod
    def is_terminal(self, game_state: TGameState) -> bool:
        """Return True if the game is in a terminal state."""

    @abstractmethod
    def reward(self, game_state: TGameState, player_id: TPlayerId) -> float | None:
        """Return the reward for the given player in the given state.

        This is typically 0, 0.5 or 1 for terminal states, depending on whether the
        player lost, drew, or won respectively.

        Reward should sum to 1 for all players.
        """

    def reward_array(self, game_state: TGameState) -> NDArray[np.float32]:
        """Return the reward for the given player in the given state as a NumPy array."""
        return np.array(
            [self.reward(game_state, player_id) for player_id in self.player_ids(game_state)], dtype=np.float32
        )

    @abstractmethod
    def pretty_str(self, game_state: TGameState) -> str:
        """Return a human-readable string representation of the game state."""
