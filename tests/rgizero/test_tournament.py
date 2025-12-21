import asyncio
import math
from typing import Sequence

import pytest
import numpy as np

from rgi.rgizero.games.base import Game
from rgi.rgizero.players.base import Player, ActionResult, TGameState, TAction
from rgi.rgizero.tournament import Tournament


# --- Mocks ---


class MockGame(Game):
    def initial_state(self):
        return 0

    def current_player_id(self, game_state):
        return 1 if game_state % 2 == 0 else 2

    def num_players(self, game_state):
        return 2

    def legal_actions(self, game_state):
        return [0, 1]

    def all_actions(self):
        return [0, 1]

    def next_state(self, game_state, action):
        return game_state + 1

    def is_terminal(self, game_state):
        return game_state >= 2

    def reward(self, game_state, player_id):
        # Deterministic winner based on some logic or just random for testing?
        # Let's say player 1 always wins if they play action 0, else player 2 wins.
        # But actions are chosen by players.
        # Let's just make it random or fixed.
        return 0.5  # Draw by default

    def reward_array(self, game_state):
        return np.array([0.5, 0.5], dtype=np.float32)

    def pretty_str(self, game_state):
        return str(game_state)


class RandomPlayer(Player):
    def select_action(self, game_state) -> ActionResult:
        return ActionResult(0, {"legal_policy": np.array([0.5, 0.5]), "legal_actions": [0, 1]})


class StrongPlayer(Player):
    def select_action(self, game_state) -> ActionResult:
        return ActionResult(0, {"legal_policy": np.array([1.0, 0.0]), "legal_actions": [0, 1]})


# --- Tests ---


def test_elo_calculation():
    game = MockGame()
    players = {"p1": lambda: RandomPlayer(), "p2": lambda: RandomPlayer()}
    tournament = Tournament(game, players, initial_elo=1000, k_factor=32)

    # Test expected score
    # Equal rating -> 0.5
    assert tournament._expected_score(1000, 1000) == 0.5
    # Higher rating -> > 0.5
    assert tournament._expected_score(1200, 1000) > 0.5

    # Test update
    # P1 wins against P2 (equal rating)
    tournament.update_elo("p1", "p2", 1.0)
    assert tournament.stats["p1"].elo == 1000 + 32 * (1.0 - 0.5)  # 1016
    assert tournament.stats["p2"].elo == 1000 + 32 * (0.0 - 0.5)  # 984

    assert tournament.stats["p1"].wins == 1
    assert tournament.stats["p2"].losses == 1
    assert tournament.stats["p1"].games_played == 1


def test_matchmaking():
    game = MockGame()
    players = {f"p{i}": lambda: RandomPlayer() for i in range(4)}
    tournament = Tournament(game, players)

    # Request 2 games
    pairings = tournament.matchmake(2)
    assert len(pairings) == 2
    for p1, p2 in pairings:
        assert p1 != p2
        assert p1 in players
        assert p2 in players


@pytest.mark.asyncio
async def test_tournament_run():
    game = MockGame()
    players = {f"p{i}": lambda: RandomPlayer() for i in range(4)}
    tournament = Tournament(game, players)

    # Run a small tournament
    await tournament.run(num_games=10, concurrent_games=2)

    total_games = sum(p.games_played for p in tournament.stats.values())
    # Each game counts for 2 players, so total_games should be 2 * num_games
    assert total_games == 20

    # Check history
    assert len(tournament.match_history) == 10
