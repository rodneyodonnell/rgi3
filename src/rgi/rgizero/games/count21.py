from dataclasses import dataclass
from typing import Sequence, TypeAlias

from typing_extensions import override

from rgi.rgizero.games.base import Game


@dataclass
class Count21State:
    score: int
    current_player: int


TGameState: TypeAlias = Count21State
TAction: TypeAlias = int
TPlayerId: TypeAlias = int


class Count21Game(Game[TGameState, TAction]):
    def __init__(self, num_players: int = 2, target: int = 21, max_guess: int = 3):
        self._num_players = num_players
        self.target = target
        self._all_actions = tuple(TAction(g + 1) for g in range(max_guess))

    @override
    def initial_state(self) -> TGameState:
        return TGameState(score=0, current_player=1)

    @override
    def current_player_id(self, game_state: TGameState) -> TPlayerId:
        return game_state.current_player

    @override
    def num_players(self, game_state: TGameState) -> int:
        return self._num_players

    @override
    def legal_actions(self, game_state: TGameState) -> Sequence[TAction]:
        return self._all_actions

    @override
    def all_actions(self) -> Sequence[TAction] | None:
        return self._all_actions

    @override
    def next_state(self, game_state: TGameState, action: TAction) -> TGameState:
        next_player = game_state.current_player % self._num_players + 1
        return TGameState(score=game_state.score + action, current_player=next_player)

    @override
    def is_terminal(self, game_state: TGameState) -> bool:
        return game_state.score >= self.target

    @override
    def reward(self, game_state: TGameState, player_id: TPlayerId) -> float:
        "Reward the player who did NOT reach the target."
        if not self.is_terminal(game_state):
            return 0.0
        return 1.0 if self.current_player_id(game_state) == player_id else -1.0

    @override
    def pretty_str(self, game_state: TGameState) -> str:
        return f"Score: {game_state.score}, Player: {game_state.current_player}"
