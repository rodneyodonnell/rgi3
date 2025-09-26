from dataclasses import dataclass
from typing import Generic, TypeVar, Sequence
from typing_extensions import override

import numpy as np

from rgi.rgizero.games.base import Game
from rgi.rgizero.common import TOKENS

TState = TypeVar("TState")
TAction = TypeVar("TAction")


@dataclass(frozen=True)
class HistoryTrackingGameState(Generic[TState, TAction]):
    base_state: TState
    action_history: tuple[TAction | str, ...]


class HistoryTrackingGame(Game[HistoryTrackingGameState[TState, TAction], TAction], Generic[TState, TAction]):
    """Game wrapper that tracks action history in the state and delegates to base game."""

    def __init__(self, base_game: Game[TState, TAction]):
        super().__init__()
        self.base_game = base_game

    @override
    def initial_state(self) -> HistoryTrackingGameState[TState, TAction]:
        base_state = self.base_game.initial_state()
        return HistoryTrackingGameState[TState, TAction](base_state, (TOKENS.START_OF_GAME,))

    @override
    def current_player_id(self, state: HistoryTrackingGameState[TState, TAction]) -> int:
        return self.base_game.current_player_id(state.base_state)

    @override
    def legal_actions(self, state: HistoryTrackingGameState[TState, TAction]) -> Sequence[TAction]:
        return self.base_game.legal_actions(state.base_state)

    @override
    def next_state(
        self, state: HistoryTrackingGameState[TState, TAction], action: TAction
    ) -> HistoryTrackingGameState[TState, TAction]:
        new_base_state = self.base_game.next_state(state.base_state, action)
        new_history = state.action_history + (action,)
        return HistoryTrackingGameState[TState, TAction](new_base_state, new_history)

    @override
    def is_terminal(self, state: HistoryTrackingGameState[TState, TAction]) -> bool:
        return self.base_game.is_terminal(state.base_state)

    @override
    def reward(self, state: HistoryTrackingGameState[TState, TAction], player_id: int) -> float:
        return self.base_game.reward(state.base_state, player_id)

    @override
    def reward_array(self, state: HistoryTrackingGameState[TState, TAction]) -> np.ndarray:
        return self.base_game.reward_array(state.base_state)

    @override
    def pretty_str(self, state: HistoryTrackingGameState[TState, TAction]) -> str:
        base_str = self.base_game.pretty_str(state.base_state)
        history_str = f"\nHistory: {list(state.action_history)}"
        return base_str + history_str

    @override
    def num_players(self, state: HistoryTrackingGameState[TState, TAction] | None = None) -> int:
        return self.base_game.num_players(state.base_state if state else None)

    @override
    def all_actions(self):
        return self.base_game.all_actions()
