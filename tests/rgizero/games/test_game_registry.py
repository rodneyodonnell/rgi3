import pytest
from rgi.rgizero.games.game_registry import create_game, list_games
from rgi.rgizero.games.connect4 import Connect4Game
from rgi.rgizero.games.count21 import Count21Game
from rgi.rgizero.games.othello import OthelloGame


def test_list_games():
    games = list_games()
    assert "connect4" in games
    assert "count21" in games
    assert "othello" in games


def test_create_game_connect4():
    game = create_game("connect4")
    assert isinstance(game.base_game, Connect4Game)


def test_create_game_count21():
    game = create_game("count21")
    assert isinstance(game.base_game, Count21Game)


def test_create_game_othello():
    game = create_game("othello")
    assert isinstance(game.base_game, OthelloGame)


def test_create_game_case_insensitive():
    game = create_game("Connect4")
    assert isinstance(game.base_game, Connect4Game)


def test_create_game_unknown():
    with pytest.raises(ValueError, match="Unknown game"):
        create_game("invalid_game_name")


def test_create_game_kwargs():
    # Connect4Game accepts connect_length
    game = create_game("connect4", connect_length=5)
    assert isinstance(game.base_game, Connect4Game)
    assert game.base_game.connect_length == 5
