from typing import Any, Type

from rgi.rgizero.games.base import Game
from rgi.rgizero.games.connect4 import Connect4Game
from rgi.rgizero.games.count21 import Count21Game
from rgi.rgizero.games.othello import OthelloGame

from rgi.rgizero.games.history_wrapper import HistoryTrackingGame

_GAME_REGISTRY: dict[str, Type[Game]] = {
    "connect4": Connect4Game,
    "count21": Count21Game,
    "othello": OthelloGame,
}


def list_games() -> list[str]:
    """List available games."""
    return sorted(list(_GAME_REGISTRY.keys()))


def create_game(name: str, **kwargs: Any) -> HistoryTrackingGame:
    """Create a game instance by name."""
    name = name.lower()
    if name not in _GAME_REGISTRY:
        raise ValueError(f"Unknown game '{name}'. Available games: {list_games()}")

    game_cls = _GAME_REGISTRY[name]
    base_game = game_cls(**kwargs)
    return HistoryTrackingGame(base_game)
