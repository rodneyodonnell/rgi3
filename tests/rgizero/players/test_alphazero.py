"""
Unit tests for AlphaZero MCTS implementation.
"""

import numpy as np
import pytest
from unittest.mock import Mock

from rgi.rgizero.games.connect4 import Connect4Game
from rgi.rgizero.players.alphazero import AlphazeroPlayer, MCTSNode


class MockEvaluator:
    """Mock evaluator for testing."""

    def __init__(self, policy_values=None, value=0.0):
        self.policy_values = policy_values or {}
        self.value = value
        self.call_count = 0

    def evaluate(self, game, state):
        self.call_count += 1
        legal_actions = list(game.legal_actions(state))

        if len(legal_actions) in self.policy_values:
            policy = self.policy_values[len(legal_actions)]
        else:
            # Default uniform policy
            policy = np.ones(len(legal_actions), dtype=np.float32) / len(legal_actions)

        # Return values for all players (assuming all players have same value for simplicity)
        num_players = game.num_players(state)
        values = np.full(num_players, self.value, dtype=np.float32)

        return policy, values


class TestMCTSNode:
    """Test MCTSNode functionality."""

    def test_node_initialization(self):
        """Test basic node initialization."""
        legal_actions = [1, 2, 3]
        policy = np.array([0.5, 0.3, 0.2])
        values = np.array([0.1, -0.1])  # 2-player values

        node = MCTSNode(legal_actions, policy, values)

        assert node.legal_actions == [1, 2, 3]
        assert np.allclose(node.prior_policy, [0.5, 0.3, 0.2])
        assert np.allclose(node.state_values, [0.1, -0.1])
        assert node.num_players == 2
        assert node.total_visits == 0
        assert len(node.children) == 3
        assert node.value_sums.shape == (3, 2)  # 3 actions x 2 players
        assert node.mean_values.shape == (3, 2)
        assert all(child is None for child in node.children)

    def test_node_initialization_size_mismatch(self):
        """Test that mismatched sizes raise an error."""
        legal_actions = [1, 2, 3]
        policy = np.array([0.5, 0.3])  # Wrong size

        with pytest.raises(AssertionError):
            MCTSNode(legal_actions, policy, np.array([0.1, -0.1]))

    def test_action_to_index(self):
        """Test action to index conversion."""
        legal_actions = [5, 10, 15]
        policy = np.array([0.3, 0.3, 0.4])
        node = MCTSNode(legal_actions, policy, np.array([0.0, 0.0]))

        assert node.action_to_index(5) == 0
        assert node.action_to_index(10) == 1
        assert node.action_to_index(15) == 2

        with pytest.raises(ValueError):
            node.action_to_index(99)

    def test_puct_values(self):
        """Test PUCT value calculation."""
        legal_actions = [1, 2]
        policy = np.array([0.7, 0.3])
        node = MCTSNode(legal_actions, policy, np.array([0.0, 0.0]))

        # No visits yet - should be driven by priors
        value1 = node.get_action_value(0, c_puct=1.0, current_player=1)
        value2 = node.get_action_value(1, c_puct=1.0, current_player=1)

        # Action 0 has higher prior, so should have higher value
        assert value1 > value2

        # Add some visits to action 1 with a high value
        node.backup(1, np.array([1.0, -1.0]))  # High positive value for player 1
        node.backup(1, np.array([1.0, -1.0]))  # Another high value

        # Now action 1 should have higher total value (Q + U)
        value1_after = node.get_action_value(0, c_puct=1.0, current_player=1)
        value2_after = node.get_action_value(1, c_puct=1.0, current_player=1)

        # Action 1 now has high Q value and should overcome the prior difference
        assert np.allclose(node.mean_values[1], [1.0, -1.0])
        # With high Q value, action 1 should now be preferred
        assert value2_after > value1_after

    def test_action_selection(self):
        """Test action selection using PUCT."""
        legal_actions = [1, 2, 3]
        policy = np.array([0.1, 0.8, 0.1])  # Strong preference for action 2
        node = MCTSNode(legal_actions, policy, np.array([0.0, 0.0]))

        # Should select action with highest prior initially
        selected_idx = node.select_action_index(c_puct=1.0, current_player=1)
        assert selected_idx == 1  # Action 2 (index 1)

    def test_backup(self):
        """Test backup/update functionality."""
        legal_actions = [1, 2]
        policy = np.array([0.5, 0.5])
        node = MCTSNode(legal_actions, policy, np.array([0.0, 0.0]))

        # Backup positive value
        node.backup(0, np.array([0.8, -0.8]))
        assert node.visit_counts[0] == 1
        assert np.allclose(node.value_sums[0], [0.8, -0.8])
        assert np.allclose(node.mean_values[0], [0.8, -0.8])
        assert node.total_visits == 1

        # Backup another value to same action
        node.backup(0, np.array([0.2, -0.2]))
        assert node.visit_counts[0] == 2
        assert np.allclose(node.value_sums[0], [1.0, -1.0])  # [0.8, -0.8] + [0.2, -0.2]
        assert np.allclose(node.mean_values[0], [0.5, -0.5])  # [1.0, -1.0] / 2
        assert node.total_visits == 2


class TestAlphazeroPlayer:
    """Test AlphazeroPlayer functionality."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        game = Connect4Game()
        evaluator = MockEvaluator()
        agent = AlphazeroPlayer(game, evaluator, simulations=100)

        assert agent.game == game
        assert agent.evaluator == evaluator
        assert agent.simulations == 100
        assert agent.c_puct == 1.0
        assert agent.temperature == 1.0

    def test_search_basic(self):
        """Test basic search functionality."""
        game = Connect4Game()
        evaluator = MockEvaluator({7: np.ones(7) / 7})  # Uniform over 7 actions
        agent = AlphazeroPlayer(game, evaluator, simulations=10)

        state = game.initial_state()
        legal_actions, visit_counts, mean_values, stats = agent.search(state)

        assert len(legal_actions) == 7
        assert len(visit_counts) == 7
        assert len(mean_values) == 7
        assert stats.simulations == 10
        assert np.sum(visit_counts) == 10  # All simulations should be counted

    def test_temperature_effects(self):
        """Test temperature effects on action selection."""
        game = Connect4Game()
        evaluator = MockEvaluator({7: np.ones(7) / 7})

        # High temperature should be more uniform
        agent_hot = AlphazeroPlayer(game, evaluator, simulations=20, temperature=2.0)
        # Low temperature should be more peaked
        agent_cold = AlphazeroPlayer(game, evaluator, simulations=20, temperature=0.1)

        state = game.initial_state()

        result_hot = agent_hot.select_action(state)
        result_cold = agent_cold.select_action(state)

        policy_hot = result_hot.info["policy"]
        policy_cold = result_cold.info["policy"]

        # Hot policy should be more uniform (higher entropy)
        entropy_hot = -np.sum(policy_hot * np.log(policy_hot + 1e-8))
        entropy_cold = -np.sum(policy_cold * np.log(policy_cold + 1e-8))

        assert entropy_hot > entropy_cold

    def test_dirichlet_noise(self):
        """Test Dirichlet noise addition."""
        game = Connect4Game()
        evaluator = MockEvaluator({7: np.array([1.0, 0, 0, 0, 0, 0, 0])})  # Concentrated policy

        agent_no_noise = AlphazeroPlayer(game, evaluator, simulations=10, add_noise=False)
        agent_with_noise = AlphazeroPlayer(game, evaluator, simulations=10, add_noise=True, noise_epsilon=0.5)

        state = game.initial_state()

        result_no_noise = agent_no_noise.select_action(state)
        result_with_noise = agent_with_noise.select_action(state)

        # With noise, the policy should be less concentrated
        policy_no_noise = result_no_noise.info["policy"]
        policy_with_noise = result_with_noise.info["policy"]

        # The max probability should be lower with noise
        assert np.max(policy_with_noise) < np.max(policy_no_noise)

    def test_value_propagation(self):
        """Test that values propagate correctly up the tree."""
        game = Connect4Game()

        # Mock evaluator that returns different values for different states
        def mock_evaluate(game_obj, state):
            # Return higher values for states with pieces in center columns
            center_pieces = np.sum(state.board[:, 2:5])
            value = 0.1 * center_pieces  # Prefer center play
            legal_actions = list(game_obj.legal_actions(state))
            policy = np.ones(len(legal_actions)) / len(legal_actions)

            # Return values for all players
            num_players = game_obj.num_players(state)
            values = np.full(num_players, value, dtype=np.float32)
            return policy, values

        evaluator = Mock()
        evaluator.evaluate = mock_evaluate

        agent = AlphazeroPlayer(game, evaluator, simulations=20)
        state = game.initial_state()

        result = agent.select_action(state)

        # With enough simulations, should show some preference for center
        # (this test is probabilistic, so we check that the system is working)
        center_actions = [3, 4, 5]  # 1-indexed center columns
        center_indices = [2, 3, 4]  # 0-indexed for mean_values array

        # Either the selected action is in center, OR center actions got reasonable visits
        center_visits = sum(result.info["visit_counts"][i] for i in center_indices)
        total_visits = sum(result.info["visit_counts"])
        center_ratio = center_visits / total_visits if total_visits > 0 else 0

        # With our biased evaluator, center should get at least some visits
        assert result.action in center_actions or center_ratio > 0.2

    def test_info_dict_contents(self):
        """Test that info dict contains expected information."""
        game = Connect4Game()
        evaluator = MockEvaluator({7: np.ones(7) / 7})
        agent = AlphazeroPlayer(game, evaluator, simulations=15, temperature=0.8)

        state = game.initial_state()
        result = agent.select_action(state)

        required_keys = {"visit_counts", "mean_values", "policy", "temperature", "legal_actions", "stats"}
        assert required_keys.issubset(result.info.keys())

        assert result.info["stats"].simulations == 15
        assert result.info["stats"].tree_depth == 1
        assert result.info["temperature"] == 0.8
        assert len(result.info["legal_actions"]) == 7
        assert len(result.info["visit_counts"]) == 7
        assert len(result.info["mean_values"]) == 7
        assert len(result.info["policy"]) == 7
        assert np.allclose(np.sum(result.info["policy"]), 1.0)  # Policy should sum to 1


class TestGameIntegration:
    """Integration tests with actual games."""

    def test_connect4_full_game(self):
        """Test playing a full Connect4 game."""
        from rgi.rgizero.players.alphazero import play_game

        game = Connect4Game()
        evaluator = MockEvaluator(
            {
                7: np.array([0.1, 0.15, 0.2, 0.3, 0.2, 0.15, 0.1]),  # Prefer center
                6: np.ones(6) / 6,
                5: np.ones(5) / 5,
                4: np.ones(4) / 4,
                3: np.ones(3) / 3,
                2: np.ones(2) / 2,
                1: np.ones(1) / 1,
            }
        )

        agents = [
            AlphazeroPlayer(game, evaluator, simulations=100, temperature=0.1),
            AlphazeroPlayer(game, evaluator, simulations=100, temperature=0.1),
        ]

        result = play_game(game, agents, max_moves=50)

        assert "winner" in result
        assert "rewards" in result
        assert "moves" in result
        assert "move_history" in result
        assert "final_state" in result

        # Game should either end in terminal state or hit move limit
        assert game.is_terminal(result["final_state"]) or result["moves"] == 50

        # If terminal, winner should be valid
        if game.is_terminal(result["final_state"]):
            assert result["winner"] in [None, 1, 2]  # None for draw

    def test_two_player_vs_multiplayer_logic(self):
        """Test that the agent behaves differently for 2-player vs multiplayer games."""
        # This is a conceptual test - we'd need a multiplayer game implementation
        # to test this properly. For now, just verify the logic exists.

        game = Connect4Game()
        evaluator = MockEvaluator({7: np.ones(7) / 7})
        agent = AlphazeroPlayer(game, evaluator, simulations=5)

        state = game.initial_state()

        # Verify it's detected as 2-player
        assert game.num_players(state) == 2

        # The agent should handle this correctly
        result = agent.select_action(state)
        assert result.action in game.legal_actions(state)

    def test_value_perspective_handling(self):
        """Test that value perspectives are handled correctly."""
        game = Connect4Game()

        # Create evaluator that returns positive values for all positions
        evaluator = MockEvaluator({7: np.ones(7) / 7}, value=0.5)
        agent = AlphazeroPlayer(game, evaluator, simulations=10)

        state = game.initial_state()

        # Run search and examine the tree
        legal_actions, visit_counts, mean_values, stats = agent.search(state)

        # With the new system, values should be consistent since we get the correct
        # player's value directly from the value array
        assert len(mean_values) == 7

        # All Q-values should be close to 0.5 since that's what we return for all players
        assert all(abs(v - 0.5) < 0.1 for v in mean_values if v != 0.0)

    def test_multiplayer_value_arrays(self):
        """Test that multiplayer value arrays work correctly."""

        class MultiplayerEvaluator:
            """Evaluator that returns different values for different players."""

            def evaluate(self, game, state):
                legal_actions = list(game.legal_actions(state))
                policy = np.ones(len(legal_actions)) / len(legal_actions)

                # Return different values for each player
                values = np.array([0.8, 0.2])  # Player 1 gets 0.8, Player 2 gets 0.2
                return policy, values

        game = Connect4Game()
        evaluator = MultiplayerEvaluator()
        agent = AlphazeroPlayer(game, evaluator, simulations=1)  # Single simulation for predictable behavior

        state = game.initial_state()

        # Player 1's turn initially
        assert game.current_player_id(state) == 1

        # Run search - this should backup Player 1's value from child states
        legal_actions, visit_counts, mean_values, stats = agent.search(state)

        # Verify that MCTS ran and backed up some value
        assert max(visit_counts) > 0, "MCTS should have visited at least one action"

        # The backed up values should be Player 1's values (0.8) from the child evaluations
        visited_indices = np.where(visit_counts > 0)[0]
        visited_q_values = mean_values[visited_indices]

        # All visited Q-values should be 0.8 (Player 1's value from child states)
        for q_val in visited_q_values:
            assert abs(q_val - 0.8) < 0.01, f"Expected ~0.8, got {q_val}"

        # Test that we're correctly indexing into the values array
        # by checking that the system doesn't confuse player indices
        assert all(abs(q - 0.8) < 0.01 for q in visited_q_values), (
            "All Q-values should be close to Player 1's value (0.8)"
        )

    def test_recursive_backup_bug(self):
        """Test that exposes the recursive backup bug."""

        class DebugEvaluator:
            """Evaluator that returns very different values per player to expose bugs."""

            def __init__(self):
                self.call_count = 0

            def evaluate(self, game, state):
                self.call_count += 1
                legal_actions = list(game.legal_actions(state))
                policy = np.ones(len(legal_actions)) / len(legal_actions)

                # Return very different values for each player to make bugs obvious
                values = np.array([0.9, 0.1])  # Player 1: 0.9, Player 2: 0.1
                return policy, values

        game = Connect4Game()
        evaluator = DebugEvaluator()

        # Use more simulations to trigger recursive case
        agent = AlphazeroPlayer(game, evaluator, simulations=10)

        state = game.initial_state()
        assert game.current_player_id(state) == 1  # Player 1's turn

        # Run search - this will hit the same action multiple times (recursive case)
        legal_actions, visit_counts, mean_values, stats = agent.search(state)

        # Find the action that was visited most (likely to have hit recursive case)
        most_visited_idx = np.argmax(visit_counts)
        most_visited_q = mean_values[most_visited_idx]

        print(f"Most visited Q-value: {most_visited_q}")
        print(f"Evaluator calls: {evaluator.call_count}")
        print(f"Visit counts: {visit_counts}")

        # With the fix, all backup values should be Player 1's value (0.9)
        # regardless of whether they come from expansion or recursion

        # The Q-value should be very close to 0.9 (Player 1's value)
        assert abs(most_visited_q - 0.9) < 0.01, (
            f"Expected Q-value ~0.9 (Player 1's value), got {most_visited_q}. "
            f"This suggests the recursive backup bug still exists."
        )

        # All simulations hit the same action, so we should have consistent backups
        assert visit_counts[most_visited_idx] == 10, "Expected all simulations to hit the same action"


if __name__ == "__main__":
    pytest.main([__file__])
