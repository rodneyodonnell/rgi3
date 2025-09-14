"""Base player interface for the framework."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from dataclasses import dataclass

TGameState = TypeVar("TGameState")
TAction = TypeVar("TAction")


@dataclass
class ActionResult(Generic[TAction]):
    """Result of a player's action selection."""

    action: TAction
    info: dict[str, Any]


class Player(ABC, Generic[TGameState, TAction]):
    """Base class for all players in the framework."""

    @abstractmethod
    def select_action(self, game_state: TGameState) -> ActionResult[TAction]:
        """Select an action from the legal actions.

        Args:
            game_state: Current game state
            legal_actions: List of legal actions

        Returns:
            ActionResult containing the selected action and optional data
        """
