from typing import Any, Optional

from rgi.rgizero.players.base import Player, TGameState, TAction, ActionResult
from rgi.rgizero.games.base import Game


class MinimaxPlayer(Player[TGameState, TAction]):
    """Minimax player with alpha-beta pruning."""

    def __init__(self, game: Game, player_id: int, max_depth: int = 4) -> None:
        self.game = game
        self.player_id = player_id
        self.max_depth = max_depth

    def select_action(self, game_state: TGameState) -> ActionResult[TAction]:
        """Select action using minimax with alpha-beta pruning."""
        legal_actions = list(self.game.legal_actions(game_state))
        eval_score, best_action = self.minimax(game_state, self.max_depth, -float("inf"), float("inf"))

        if best_action is None:
            best_action = legal_actions[0]

        return ActionResult(best_action, {"eval_score": eval_score})

    def evaluate(self, state) -> float:
        """Evaluate state from this player's perspective."""
        if self.game.is_terminal(state):
            return self.game.reward(state, self.player_id)
        return self.heuristic(state)

    def heuristic(self, state) -> float:
        """Heuristic evaluation for non-terminal states.

        Override this method to provide game-specific heuristics.
        Default implementation returns 0 (neutral evaluation).
        """
        return 0.0

    def minimax(self, state, depth: int, alpha: float, beta: float) -> tuple[float, Optional[Any]]:
        """Minimax search with alpha-beta pruning."""
        if self.game.is_terminal(state) or depth == 0:
            return self.evaluate(state), None

        current_player = self.game.current_player_id(state)
        is_maximizing_player = current_player == self.player_id

        legal_actions = list(self.game.legal_actions(state))
        best_action = None

        if is_maximizing_player:
            max_eval = -float("inf")
            for action in legal_actions:
                next_state = self.game.next_state(state, action)
                eval_score, _ = self.minimax(next_state, depth - 1, alpha, beta)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cut-off
            return max_eval, best_action
        else:
            min_eval = float("inf")
            for action in legal_actions:
                next_state = self.game.next_state(state, action)
                eval_score, _ = self.minimax(next_state, depth - 1, alpha, beta)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval, best_action
