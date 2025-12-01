import random

from rgi.rgizero.players.base import Player, TGameState, TAction, ActionResult
from rgi.rgizero.games.base import Game


class RandomPlayer(Player[TGameState, TAction]):
    """Random player that selects actions uniformly at random."""

    def __init__(self, game: Game, seed: int | None = None) -> None:
        self.game = game
        self.rng = random.Random(seed)

    def select_action(self, game_state: TGameState) -> ActionResult[TAction]:
        """Select action randomly from legal actions."""
        legal_actions = self.game.legal_actions(game_state)
        action = self.rng.choice(legal_actions)
        
        # Create uniform policy
        num_legal = len(legal_actions)
        legal_policy = [1.0 / num_legal] * num_legal
        
        return ActionResult(action, {
            "legal_policy": legal_policy,
            "legal_actions": legal_actions
        })
