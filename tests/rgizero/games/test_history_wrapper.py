import pytest
import random

from rgi.rgizero.common import TOKENS
from rgi.rgizero.games.count21 import Count21Game
from rgi.rgizero.games.history_wrapper import HistoryTrackingGame


@pytest.fixture
def base_game():
    # Use Count21Game as a simple test game.
    return Count21Game()


@pytest.fixture
def wrapped_game(base_game):
    return HistoryTrackingGame(base_game)


def test_initial_state(base_game: Count21Game, wrapped_game: HistoryTrackingGame):
    base_state = base_game.initial_state()
    wrapped_state = wrapped_game.initial_state()
    assert wrapped_state.base_state == base_state
    assert wrapped_state.action_history == (TOKENS.START_OF_GAME,)
    assert wrapped_game.current_player_id(wrapped_state) == base_game.current_player_id(base_state)


def test_legal_actions(base_game: Count21Game, wrapped_game: HistoryTrackingGame):
    base_state = base_game.initial_state()
    wrapped_state = wrapped_game.initial_state()
    assert wrapped_game.legal_actions(wrapped_state) == base_game.legal_actions(base_state)


def test_play_full_wrapped_game(base_game: Count21Game, wrapped_game: HistoryTrackingGame):
    rng = random.Random(42)
    base_state = base_game.initial_state()
    wrapped_state = wrapped_game.initial_state()
    while True:
        assert wrapped_game.is_terminal(wrapped_state) == base_game.is_terminal(base_state)
        assert wrapped_state.base_state == base_state
        assert wrapped_game.current_player_id(wrapped_state) == base_game.current_player_id(base_state)
        assert wrapped_game.legal_actions(wrapped_state) == base_game.legal_actions(base_state)
        assert wrapped_game.all_actions() == base_game.all_actions()
        assert wrapped_game.num_players(wrapped_state) == base_game.num_players(base_state)
        assert wrapped_game.reward(wrapped_state, 1) == base_game.reward(base_state, 1)

        if wrapped_game.is_terminal(wrapped_state):
            assert (wrapped_game.reward_array(wrapped_state) == base_game.reward_array(base_state)).all()
            break
        legal_actions = wrapped_game.legal_actions(wrapped_state)
        action = rng.choice(legal_actions)
        wrapped_state = wrapped_game.next_state(wrapped_state, action)
        base_state = base_game.next_state(base_state, action)
