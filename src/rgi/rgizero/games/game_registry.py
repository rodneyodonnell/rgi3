from typing import Any, Type

from rgi.rgizero.games.base import Game
from rgi.rgizero.games.connect4 import Connect4Game
from rgi.rgizero.games.count21 import Count21Game
from rgi.rgizero.games.othello import OthelloGame


_GAME_REGISTRY: dict[str, Type[Game]] = {
    "connect4": Connect4Game,
    "count21": Count21Game,
    "othello": OthelloGame,
}


def list_games() -> list[str]:
    """List available games."""
    return sorted(list(_GAME_REGISTRY.keys()))


def create_game(name: str, **kwargs: Any) -> Game:
    """Create a game instance by name.

    Args:
        name: Name of the game (e.g. "connect4")
        **kwargs: Arguments passed to the game constructor.

    Returns:
        Game instance.

    Raises:
        ValueError: If game name is unknown.
    """
    name = name.lower()
    if name not in _GAME_REGISTRY:
        raise ValueError(f"Unknown game '{name}'. Available games: {list_games()}")
    
    game_cls = _GAME_REGISTRY[name]
    return game_cls(**kwargs)
