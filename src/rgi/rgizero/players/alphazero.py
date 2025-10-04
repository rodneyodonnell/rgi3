"""
AlphaZero-style Monte Carlo Tree Search implementation.

Simple design using parallel arrays for actions, visit counts, and values.
Works with dynamic action spaces by storing legal_actions alongside statistics.

Algorithm Overview:
1. Selection: Use PUCT formula to traverse from root to leaf
2. Expansion: Add new node and evaluate with neural network
3. Backup: Propagate values up the tree, flipping signs for alternating players
4. Action Selection: Choose action based on visit counts
"""

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from rgi.rgizero.games.base import Game
from rgi.rgizero.players.base import Player, TGameState, TAction, ActionResult


@dataclass
class NetworkEvaluatorResult:
    """Result of a network evaluation.

    - policy: Array of policy probabilities for legal actions, shape: (num_legal_actions,)
    - values: Array of value estimates for each player, in range (0, 1) and sum to 1, shape: (num_players,)
    """

    policy: np.ndarray
    values: np.ndarray


class NetworkEvaluator(Protocol):
    """Protocol for neural network evaluation of game positions."""

    def evaluate(self, game: Game, state, legal_actions: list[Any]) -> NetworkEvaluatorResult:
        """Evaluate a game state.

        Args:
            game: The game instance
            state: The game state to evaluate

        Returns NetworkEvaluatorResult
        """
        ...


@dataclass
class MCTSStats:
    """Statistics from MCTS search."""

    simulations: int = 0
    tree_depth: int = 0


class MCTSNode:
    """A node in the MCTS tree.

    N(s)     : total visits to state s = sum_b N(s,b)
    N(s,a)   : visits to action a at state s
    W(s,a)   : total value accumulated through (s,a)
    Q(s,a)   : mean action value = W(s,a) / N(s,a)
    P(s,a)   : prior probability of action a from the policy net
    c_puct   : exploration constant (hyperparameter)
    """

    def __init__(
        self,
        legal_actions: Sequence,
        prior_policy: np.ndarray,
        state_values: np.ndarray,
    ):
        """Initialize a new MCTS node.

        Args:
            legal_actions: List of legal actions from this state
            prior_policy: Prior policy probabilities (same length as legal_actions)
            state_values: Value estimates for all players [player1_value, player2_value, ...]
        """
        assert len(legal_actions) == len(prior_policy)

        self.legal_actions = list(legal_actions)  # Actions available from this state
        self.prior_policy = prior_policy.copy()  # P(s,a) - prior probabilities
        self.state_values = state_values.copy()  # Initial value estimates for all players
        self.num_players = len(state_values)

        # Visit statistics, indexed by legal_action.
        num_actions = len(legal_actions)
        self.visit_counts = np.zeros(num_actions, dtype=np.int32)  # N(s,a)
        self.value_sums = np.zeros((num_actions, self.num_players), dtype=np.float32)  # W(s,a,p)
        self.mean_values = np.zeros((num_actions, self.num_players), dtype=np.float32)  # Q(s,a,p)

        # Child nodes - indexed same as actions
        self.children: list[MCTSNode | None] = [None] * len(legal_actions)

        # Total visits to this node
        self.total_visits = 0

    def action_to_index(self, action) -> int:
        """Find index of action in legal_actions."""
        try:
            return self.legal_actions.index(action)
        except ValueError:
            raise ValueError(f"Action {action} not found in legal actions {self.legal_actions}")

    def is_expanded(self, action_idx: int) -> bool:
        """Check if an action has been expanded (has a child node)."""
        return self.children[action_idx] is not None

    def get_action_value(self, action_idx: int, c_puct: float, current_player: int) -> float:
        """Calculate PUCT value for an action.

        Args:
            action_idx: Index of the action in legal_actions
            c_puct: PUCT exploration constant
            current_player: Current player (1-based)

        Returns:
            PUCT value combining exploitation (Q) and exploration (U)
        """
        # Exploitation: Q(s,a) for current player
        current_player_idx = current_player - 1  # Convert to 0-based
        q_value = self.mean_values[action_idx, current_player_idx]

        # Exploration: U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        exploration_factor = math.sqrt(max(1, self.total_visits)) / (1 + self.visit_counts[action_idx])
        exploration = c_puct * self.prior_policy[action_idx] * exploration_factor

        return q_value + exploration

    def select_action_index(self, c_puct: float, current_player: int) -> int:
        """Select best action index using PUCT rule.

        Args:
            c_puct: PUCT exploration constant
            current_player: Current player (1-based)

        Returns:
            Index of selected action in legal_actions
        """
        best_value = float("-inf")
        best_idx = 0

        for action_idx in range(len(self.legal_actions)):
            value = self.get_action_value(action_idx, c_puct, current_player)
            if value > best_value:
                best_value = value
                best_idx = action_idx

        return best_idx

    def backup(self, action_idx: int, values: np.ndarray):
        """Update statistics after a simulation.

        Args:
            action_idx: Index of action that was taken
            values: Value array for all players [player1_value, player2_value, ...]
        """
        assert len(values) == self.num_players
        self.visit_counts[action_idx] += 1
        self.value_sums[action_idx] += values
        self.mean_values[action_idx] = self.value_sums[action_idx] / self.visit_counts[action_idx]
        self.total_visits += 1


@dataclass
class SearchResult:
    """Result of a search."""

    legal_actions: list[Any]  # legal actions, shape: (num_actions,)
    visit_counts: np.ndarray  # visit count for next action, shape: (num_actions,)
    mean_values: np.ndarray  # mean values for current player, shape: (num_actions,)
    all_mean_values: np.ndarray  # mean values for all players, shape: (num_players, num_actions)
    stats: MCTSStats


class AlphazeroPlayer(Player[TGameState, TAction]):
    """AlphaZero-style MCTS agent using neural network evaluation."""

    def __init__(
        self,
        game: Game,
        evaluator: NetworkEvaluator,
        simulations: int = 800,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        add_noise: bool = False,
        noise_alpha: float = 0.3,
        noise_epsilon: float = 0.25,
        rng: np.random.Generator | None = None,
    ):
        """Initialize MCTS agent.

        Args:
            game: Game instance that implements the Game interface
            evaluator: Neural network evaluator
            simulations: Number of MCTS simulations per move
            c_puct: PUCT exploration constant
            temperature: Temperature for action selection (0 = greedy, >0 = stochastic)
            add_noise: Whether to add Dirichlet noise to root prior
            noise_alpha: Alpha parameter for Dirichlet noise
            noise_epsilon: Mixing ratio for Dirichlet noise
            rng: Optional generator for reproducible randomness
        """
        self.game = game
        self.evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.add_noise = add_noise
        self.noise_alpha = noise_alpha
        self.noise_epsilon = noise_epsilon
        self.rng = rng or np.random.default_rng()

    def search(self, root_state) -> SearchResult:
        """Run MCTS search from root state.

        Args:
            root_state: Game state to search from

        Returns:
            Tuple of (legal_actions, visit_counts, mean_values, stats)
        """
        stats = MCTSStats()

        # Get legal actions and evaluate root state
        legal_actions = self.game.legal_actions(root_state)
        result = self.evaluator.evaluate(self.game, root_state, legal_actions)
        policy = result.policy
        values = result.values

        # Add Dirichlet noise to root if enabled
        if self.add_noise:
            noise = self.rng.dirichlet([self.noise_alpha] * len(legal_actions))
            policy = (1 - self.noise_epsilon) * policy + self.noise_epsilon * noise
            policy = policy / np.sum(policy)

        root_node = MCTSNode(legal_actions, policy, values)

        # Run simulations
        for _ in range(self.simulations):
            self._simulate(root_state, root_node, stats)
            stats.simulations += 1

        # Return statistics from current player's perspective
        current_player = self.game.current_player_id(root_state)
        current_player_idx = current_player - 1
        all_mean_values = root_node.mean_values.copy()
        mean_values = all_mean_values[:, current_player_idx]

        return SearchResult(
            legal_actions, root_node.visit_counts.astype(np.float32), mean_values, all_mean_values, stats
        )

    def _simulate(self, state, node: MCTSNode, stats: MCTSStats, depth: int = 0):
        """Run a single MCTS simulation.

        Args:
            state: Current game state
            node: Current MCTS node
            stats: Statistics tracker
            depth: Current search depth

        Returns:
            Value array for all players
        """
        stats.tree_depth = max(stats.tree_depth, depth)

        # Terminal state - return actual game values
        if self.game.is_terminal(state):
            return self.game.reward_array(state)

        if not node.legal_actions:
            raise ValueError("No legal actions in non-terminal state")

        # Selection: choose action using PUCT
        current_player = self.game.current_player_id(state)
        action_idx = node.select_action_index(self.c_puct, current_player)
        action = node.legal_actions[action_idx]

        # Get next state
        next_state = self.game.next_state(state, action)

        # Check if child node exists
        if not node.is_expanded(action_idx):
            # Expansion: create new child node
            legal_actions = self.game.legal_actions(next_state)
            child_result = self.evaluator.evaluate(self.game, next_state, legal_actions)

            # Create child node with full value array
            child_node = MCTSNode(legal_actions, child_result.policy, child_result.values)
            node.children[action_idx] = child_node

            # Backup full value array
            node.backup(action_idx, child_result.values)
            return child_result.values
        else:
            # Recursion: continue search in existing child
            child_node = node.children[action_idx]
            values = self._simulate(next_state, child_node, stats, depth + 1)

            # Backup the full value array - no conversion needed
            node.backup(action_idx, values)
            return values

    def select_action(self, game_state: Any) -> ActionResult[Any]:
        """Player interface implementation."""
        search_result = self.search(game_state)

        # Convert visit counts to policy using temperature
        if self.temperature == 0:
            # Deterministic selection
            action_idx = int(np.argmax(search_result.visit_counts))
            policy = np.zeros_like(search_result.visit_counts, dtype=np.float32)
            policy[action_idx] = 1.0
        else:
            # Stochastic selection with temperature
            epsilon = 1e-10
            log_counts = np.log(search_result.visit_counts + epsilon)
            log_policy = log_counts / (self.temperature + epsilon)
            stable_log_policy = log_policy - np.max(log_policy)
            unnormalised_policy = np.exp(stable_log_policy)
            policy = unnormalised_policy / np.sum(unnormalised_policy)
            action_idx = self.rng.choice(len(policy), p=policy)

        action = search_result.legal_actions[action_idx]
        return ActionResult(
            action,
            {
                "visits": search_result.visit_counts[action_idx],
                "visit_counts": search_result.visit_counts,
                "mean_value": search_result.mean_values[action_idx],
                "mean_values": search_result.mean_values,
                "policy": policy,
                "temperature": self.temperature,
                "legal_actions": search_result.legal_actions,
                "stats": search_result.stats,
            },
        )


def play_game(game: Game, agents: list[Player[TGameState, TAction]], max_actions: int = 1000) -> dict:
    """Play a complete game between MCTS agents.

    Args:
        game: Game instance
        agents: List of MCTS agents (one per player)
        max_moves: Maximum number of moves before declaring draw

    Returns:
        Dictionary with game outcome and statistics
    """
    state = game.initial_state()
    action_history = []
    policies = []
    legal_action_idx_list = []

    all_actions = game.all_actions()
    all_action_idx_map = {action: idx for idx, action in enumerate(all_actions)}

    num_actions = 0
    while not game.is_terminal(state) and num_actions < max_actions:
        current_player = game.current_player_id(state)
        agent = agents[current_player - 1]  # Convert to 0-based indexing

        result = agent.select_action(state)
        action_history.append(result.action)
        policies.append(result.info["policy"])
        legal_actions = result.info["legal_actions"]
        legal_action_idx = np.array([all_action_idx_map[action] for action in legal_actions])
        legal_action_idx_list.append(legal_action_idx)

        state = game.next_state(state, result.action)
        num_actions += 1

    # Determine outcome
    rewards = game.reward_array(state)
    if game.is_terminal(state):
        winner = None
        rewards = game.reward_array(state)
        for player_id, reward in zip(game.player_ids(state), rewards):
            if reward >= 1.0:
                winner = player_id
    else:
        # Max moves reached - declare draw
        winner = None

    return {
        "winner": winner,
        "rewards": rewards,
        "action_history": action_history,
        "policies": policies,
        "final_state": state,
        "legal_action_idx": legal_action_idx_list,
    }
