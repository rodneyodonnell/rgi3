from typing import Sequence

from rgi.rgizero.players.base import Player, TGameState, TAction, ActionResult
from rgi.rgizero.games.base import Game

_INDEX_PREFIX = "i:"


class HumanPlayer(Player[TGameState, TAction]):
    """Human player that prompts for input via console."""

    def __init__(self, game: Game) -> None:
        self.game = game

    def select_action(self, game_state: TGameState) -> ActionResult[TAction]:
        """Select action via human input."""
        legal_actions = list(self.game.legal_actions(game_state))
        action = self._select_action_from_user(game_state, legal_actions)
        return ActionResult(action, {})

    def _select_action_from_user(self, game_state: TGameState, legal_actions: Sequence[TAction]) -> TAction:
        """Prompt user for action selection."""
        while True:
            print("\nCurrent game state:")
            print(self.game.pretty_str(game_state))
            print(f"\nPlayer {self.game.current_player_id(game_state)}'s turn")
            print("Legal actions:")
            for i, action in enumerate(legal_actions):
                print(f"  {_INDEX_PREFIX}{i + 1} or {action}")

            choice_str = input("\nEnter the index of your chosen action: ").strip()

            # i:x is verbose & safer way of choosing an action
            if choice_str.startswith(_INDEX_PREFIX):
                try:
                    idx = int(choice_str[len(_INDEX_PREFIX) :]) - 1
                    return legal_actions[idx]
                except (ValueError, IndexError):
                    pass

            # User can type the action directly
            for action in legal_actions:
                if str(action) == choice_str:
                    return action

            # User can choose the action number without the i: prefix
            try:
                idx = int(choice_str) - 1
                return legal_actions[idx]
            except (ValueError, IndexError):
                pass

            print("##\n##\n## Invalid input. Please enter a valid action.\n##\n##")
