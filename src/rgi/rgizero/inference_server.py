"""
Inference Server for parallel self-play.

Architecture:
- One InferenceServer process handles GPU inference
- Multiple GameWorker processes play games using MCTS
- Workers send eval requests via multiprocessing queues
- Server batches requests and returns results

This allows true parallel game execution with proper GPU batching.
"""

import asyncio
import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Any, Optional
import queue

import numpy as np

from rgi.rgizero.data.trajectory_dataset import Vocab
from rgi.rgizero.games import game_registry
from rgi.rgizero.games.base import Game
from rgi.rgizero.models.action_history_transformer import ActionHistoryTransformer
from rgi.rgizero.evaluators import ActionHistoryTransformerEvaluator
from rgi.rgizero.players.alphazero import (
    NetworkEvaluator,
    NetworkEvaluatorResult,
    MCTSNode,
    MCTSStats,
    SearchResult,
)
from rgi.rgizero.players.base import Player, TGameState, TAction, ActionResult


class SyncAlphazeroPlayer(Player[TGameState, TAction]):
    """
    Synchronous AlphaZero MCTS player for use in worker processes.

    Unlike AlphazeroPlayer, this uses synchronous evaluate() only,
    avoiding the overhead of asyncio event loops in each worker.
    """

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
        self.game = game
        self.evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.add_noise = add_noise
        self.noise_alpha = noise_alpha
        self.noise_epsilon = noise_epsilon
        self.rng = rng or np.random.default_rng()

    def _create_root_node(self, root_state: TGameState) -> MCTSNode:
        legal_actions = self.game.legal_actions(root_state)
        result = self.evaluator.evaluate(self.game, root_state, legal_actions)
        return MCTSNode(legal_actions, result.legal_policy, result.player_values)

    def _simulate(self, state, node: MCTSNode, stats: MCTSStats, depth: int = 0) -> np.ndarray:
        """Run a single MCTS simulation synchronously."""
        stats.tree_depth = max(stats.tree_depth, depth)

        if self.game.is_terminal(state):
            reward_array = self.game.reward_array(state)
            return (reward_array.shape[0] * reward_array - 1) / reward_array.shape[0]

        if not node.legal_actions:
            raise ValueError("No legal actions in non-terminal state")

        current_player = self.game.current_player_id(state)
        action_idx = node.select_action_index(self.c_puct, current_player)
        action = node.legal_actions[action_idx]
        next_state = self.game.next_state(state, action)

        if not node.is_expanded(action_idx):
            # Expansion
            legal_actions = self.game.legal_actions(next_state)
            child_result = self.evaluator.evaluate(self.game, next_state, legal_actions)
            child_node = MCTSNode(legal_actions, child_result.legal_policy, child_result.player_values)
            node.children[action_idx] = child_node
            node.backup(action_idx, child_result.player_values)
            return child_result.player_values
        else:
            # Recurse
            child_node = node.children[action_idx]
            player_values = self._simulate(next_state, child_node, stats, depth + 1)
            node.backup(action_idx, player_values)
            return player_values

    def search(self, root_state) -> SearchResult:
        """Run MCTS search synchronously."""
        stats = MCTSStats()
        root_node = self._create_root_node(root_state)

        if self.add_noise:
            noise = self.rng.dirichlet([self.noise_alpha] * len(root_node.prior_legal_policy))
            unnormalised = (1 - self.noise_epsilon) * root_node.prior_legal_policy + self.noise_epsilon * noise
            root_node.prior_legal_policy = unnormalised / np.sum(unnormalised)

        for _ in range(self.simulations):
            self._simulate(root_state, root_node, stats)
            stats.simulations += 1

        current_player = self.game.current_player_id(root_state)
        current_player_idx = current_player - 1
        all_players_mean_values = root_node.mean_player_values.copy()
        current_player_mean_values = all_players_mean_values[:, current_player_idx]

        return SearchResult(
            root_node.legal_actions,
            root_node.legal_action_visit_counts.astype(np.float32),
            current_player_mean_values,
            all_players_mean_values,
            stats,
            root_node,
        )

    def select_action(self, game_state: Any) -> ActionResult[Any]:
        """Select action synchronously."""
        search_result = self.search(game_state)

        if self.temperature == 0:
            action_idx = int(np.argmax(search_result.legal_action_visit_counts))
            legal_policy = np.zeros_like(search_result.legal_action_visit_counts, dtype=np.float32)
            legal_policy[action_idx] = 1.0
        else:
            epsilon = 1e-10
            log_counts = np.log(search_result.legal_action_visit_counts + epsilon)
            log_policy = log_counts / (self.temperature + epsilon)
            stable_log_policy = log_policy - np.max(log_policy)
            unnormalised = np.exp(stable_log_policy)
            legal_policy = unnormalised / np.sum(unnormalised)
            action_idx = self.rng.choice(len(legal_policy), p=legal_policy)

        action = search_result.legal_actions[action_idx]
        return ActionResult(
            action,
            {
                "legal_action_visit_counts": search_result.legal_action_visit_counts,
                "current_player_mean_values": search_result.current_player_mean_values,
                "legal_policy": legal_policy,
                "temperature": self.temperature,
                "legal_actions": search_result.legal_actions,
                "stats": search_result.stats,
            },
        )


@dataclass
class EvalRequest:
    """Request for neural network evaluation."""

    worker_id: int
    request_id: int
    game_name: str
    action_history: list
    legal_actions: list


@dataclass
class EvalResponse:
    """Response from neural network evaluation."""

    request_id: int
    legal_policy: np.ndarray  # Probabilities over legal actions
    player_values: np.ndarray  # Expected value for each player


class InferenceServer:
    """
    GPU inference server that batches requests from multiple workers.

    Runs in its own process, receives eval requests via queue,
    batches them together, runs inference, and sends results back.
    """

    def __init__(
        self,
        model: ActionHistoryTransformer,
        vocab: Vocab,
        device: str,
        max_batch_size: int = 1024,
        max_wait_ms: float = 1.0,
        verbose: bool = False,
    ):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.verbose = verbose

        # Create base evaluator for batch inference
        self.evaluator = ActionHistoryTransformerEvaluator(
            model=model,
            device=device,
            block_size=100,  # TODO: make configurable
            vocab=vocab,
            verbose=verbose,
        )

        # Queues will be set by start()
        self.request_queue: Optional[mp.Queue] = None
        self.response_queues: dict[int, mp.Queue] = {}

        # Stats
        self.total_batches = 0
        self.total_evals = 0
        self.total_time = 0.0

    def run(self, request_queue: mp.Queue, response_queues: dict[int, mp.Queue], stop_event: mp.Event):
        """
        Main loop - runs in inference server process.

        Collects requests, batches them, runs inference, returns results.
        """
        self.request_queue = request_queue
        self.response_queues = response_queues

        if self.verbose:
            print(f"InferenceServer starting on device {self.device}")

        while not stop_event.is_set():
            batch = self._collect_batch()
            if batch:
                self._process_batch(batch)

        if self.verbose:
            print(f"InferenceServer stopping. Stats: {self.total_batches} batches, {self.total_evals} evals")

    def _collect_batch(self) -> list[EvalRequest]:
        """Collect a batch of requests from the queue."""
        batch = []
        deadline = time.perf_counter() + self.max_wait_ms / 1000

        # Block for first request (with timeout to check stop_event)
        try:
            req = self.request_queue.get(timeout=0.1)
            batch.append(req)
        except queue.Empty:
            return []

        # Collect more requests until batch full or deadline
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                req = self.request_queue.get(timeout=remaining)
                batch.append(req)
            except queue.Empty:
                break

        return batch

    def _process_batch(self, batch: list[EvalRequest]):
        """Process a batch of requests and send responses."""
        t0 = time.perf_counter()

        # TODO: Currently assumes all requests are for same game
        # Could group by game_name if needed
        game_name = batch[0].game_name
        game = game_registry.create_game(game_name)

        # Create fake states with action histories
        # The evaluator expects game states, but we only have action histories
        # We need to reconstruct states or modify the evaluator
        states = []
        legal_actions_list = []

        for req in batch:
            # Create a minimal state object with action_history
            state = _ActionHistoryState(req.action_history)
            states.append(state)
            legal_actions_list.append(req.legal_actions)

        # Run batch evaluation
        results = self.evaluator.evaluate_batch(game, states, legal_actions_list)

        # Send responses
        for req, result in zip(batch, results):
            response = EvalResponse(
                request_id=req.request_id,
                legal_policy=result.legal_policy,
                player_values=result.player_values,
            )
            self.response_queues[req.worker_id].put(response)

        # Update stats
        elapsed = time.perf_counter() - t0
        self.total_batches += 1
        self.total_evals += len(batch)
        self.total_time += elapsed

        if self.verbose and self.total_batches % 100 == 0:
            mean_batch = self.total_evals / self.total_batches
            mean_evals_per_sec = self.total_evals / self.total_time
            print(
                f"InferenceServer: {self.total_batches} batches, mean_size={mean_batch:.1f}, evals/sec={mean_evals_per_sec:.1f}"
            )


class _ActionHistoryState:
    """Minimal state object that just wraps action history."""

    def __init__(self, action_history: list):
        self.action_history = action_history


class IPCNetworkEvaluator:
    """
    Network evaluator that sends requests to InferenceServer via IPC.

    Used by GameWorker processes to get neural network evaluations.
    """

    def __init__(
        self,
        worker_id: int,
        game_name: str,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
    ):
        self.worker_id = worker_id
        self.game_name = game_name
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.next_request_id = 0

    def evaluate(self, game, state, legal_actions):
        """Send eval request to server and wait for response."""
        request_id = self.next_request_id
        self.next_request_id += 1

        # Send request
        req = EvalRequest(
            worker_id=self.worker_id,
            request_id=request_id,
            game_name=self.game_name,
            action_history=list(state.action_history),
            legal_actions=list(legal_actions),
        )
        self.request_queue.put(req)

        # Wait for response
        while True:
            response = self.response_queue.get()
            if response.request_id == request_id:
                return NetworkEvaluatorResult(
                    legal_policy=response.legal_policy,
                    player_values=response.player_values,
                )
            else:
                # Wrong response, put it back (shouldn't happen with proper design)
                self.response_queue.put(response)


def play_game_sync(game: Game, agents: list, max_actions: int = 1000) -> dict:
    """Play a complete game synchronously. Simplified version of play_game."""
    state = game.initial_state()
    action_history = []
    legal_policies = []
    legal_action_idx_list = []

    all_actions = game.all_actions()
    all_action_idx_map = {action: idx for idx, action in enumerate(all_actions)}

    num_actions = 0
    while not game.is_terminal(state) and num_actions < max_actions:
        current_player = game.current_player_id(state)
        agent = agents[current_player - 1]

        result = agent.select_action(state)
        action_history.append(result.action)
        legal_policies.append(result.info["legal_policy"])
        legal_actions = result.info["legal_actions"]
        legal_action_idx = np.array([all_action_idx_map[action] for action in legal_actions])
        legal_action_idx_list.append(legal_action_idx)

        state = game.next_state(state, result.action)
        num_actions += 1

    rewards = game.reward_array(state)
    winner = None
    if game.is_terminal(state):
        for player_id, reward in zip(game.player_ids(state), rewards):
            if reward >= 1.0:
                winner = player_id

    return {
        "winner": winner,
        "rewards": rewards,
        "action_history": action_history,
        "legal_policies": legal_policies,
        "final_state": state,
        "legal_action_idx": legal_action_idx_list,
    }


def game_worker_process(
    worker_id: int,
    game_name: str,
    num_games: int,
    num_simulations: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    seed: int,
):
    """
    Worker process that plays games and sends trajectories to result queue.

    This runs in its own process, using IPC to get neural network evaluations.
    """
    # Setup
    game = game_registry.create_game(game_name)
    rng = np.random.default_rng(seed)

    # Create IPC-based evaluator
    evaluator = IPCNetworkEvaluator(
        worker_id=worker_id,
        game_name=game_name,
        request_queue=request_queue,
        response_queue=response_queue,
    )

    trajectories = []

    for game_idx in range(num_games):
        # Create sync player for this game
        player = SyncAlphazeroPlayer(
            game=game,
            evaluator=evaluator,
            rng=rng,
            add_noise=True,
            simulations=num_simulations,
        )

        # Play game synchronously
        trajectory = play_game_sync(game, [player, player])
        trajectories.append(trajectory)

        if (game_idx + 1) % 10 == 0:
            print(f"Worker {worker_id}: {game_idx + 1}/{num_games} games")

    # Send results
    result_queue.put((worker_id, trajectories))


class ParallelSelfPlay:
    """
    Orchestrates parallel self-play using inference server + worker processes.
    """

    def __init__(
        self,
        model: ActionHistoryTransformer,
        game_name: str,
        vocab: Vocab,
        device: str,
        num_workers: int = 4,
        num_simulations: int = 50,
        verbose: bool = False,
    ):
        self.model = model
        self.game_name = game_name
        self.vocab = vocab
        self.device = device
        self.num_workers = num_workers
        self.num_simulations = num_simulations
        self.verbose = verbose

    def play_games(self, total_games: int, seed: int = 42) -> list:
        """
        Play games in parallel using inference server + workers.

        Returns list of trajectories.
        """
        # Divide games among workers
        games_per_worker = total_games // self.num_workers
        remainder = total_games % self.num_workers

        # Create queues
        request_queue = mp.Queue()
        response_queues = {i: mp.Queue() for i in range(self.num_workers)}
        result_queue = mp.Queue()
        stop_event = mp.Event()

        # Create inference server (runs in main process for now, could be separate)
        server = InferenceServer(
            model=self.model,
            vocab=self.vocab,
            device=self.device,
            verbose=self.verbose,
        )

        # Start worker processes
        workers = []
        rng = np.random.default_rng(seed)

        for i in range(self.num_workers):
            num_games = games_per_worker + (1 if i < remainder else 0)
            worker_seed = int(rng.integers(0, 2**31))

            p = mp.Process(
                target=game_worker_process,
                args=(
                    i,
                    self.game_name,
                    num_games,
                    self.num_simulations,
                    request_queue,
                    response_queues[i],
                    result_queue,
                    worker_seed,
                ),
            )
            workers.append(p)
            p.start()

        # Run inference server in main thread until all workers done
        # TODO: Run server in separate process for true parallelism
        import threading

        server_thread = threading.Thread(
            target=server.run,
            args=(request_queue, response_queues, stop_event),
        )
        server_thread.start()

        # Collect results
        all_trajectories = []
        for _ in range(self.num_workers):
            worker_id, trajectories = result_queue.get()
            all_trajectories.extend(trajectories)
            if self.verbose:
                print(f"Collected {len(trajectories)} trajectories from worker {worker_id}")

        # Stop server
        stop_event.set()
        server_thread.join(timeout=5.0)

        # Wait for workers
        for p in workers:
            p.join(timeout=5.0)

        return all_trajectories


# =============================================================================
# HYBRID APPROACH: Async workers with batch IPC
# =============================================================================


@dataclass
class BatchEvalRequest:
    """Batch request for neural network evaluation."""

    worker_id: int
    batch_id: int
    game_name: str
    action_histories: list[list]  # List of action histories
    legal_actions_list: list[list]  # List of legal actions per state


@dataclass
class BatchEvalResponse:
    """Batch response from neural network evaluation."""

    batch_id: int
    legal_policies: list[np.ndarray]  # List of policies
    player_values_list: list[np.ndarray]  # List of values


class BatchInferenceServer:
    """
    GPU inference server that receives BATCHES from workers.

    Workers send batches of 50-100 requests.
    Server merges batches from all workers, runs mega-batch on GPU.
    """

    def __init__(
        self,
        model: ActionHistoryTransformer,
        vocab: Vocab,
        device: str,
        max_batch_size: int = 2048,
        max_wait_ms: float = 2.0,
        verbose: bool = False,
    ):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.verbose = verbose

        self.evaluator = ActionHistoryTransformerEvaluator(
            model=model,
            device=device,
            block_size=100,
            vocab=vocab,
            verbose=verbose,
        )

        self.total_batches = 0
        self.total_evals = 0
        self.total_time = 0.0

    def run(self, request_queue: mp.Queue, response_queues: dict[int, mp.Queue], stop_event: mp.Event):
        """Main loop - collects batch requests, merges, runs inference."""
        if self.verbose:
            print(f"BatchInferenceServer starting on device {self.device}")

        # Pipelined approach: collect first batch, then while GPU runs, collect next batch
        pending_batch = []

        while not stop_event.is_set():
            # Get first request if we have none pending
            if not pending_batch:
                try:
                    req = request_queue.get(timeout=0.1)
                    pending_batch.append(req)
                except queue.Empty:
                    continue

            # Immediately grab any more requests that are already queued
            while len(pending_batch) < 50:  # Limit batch assembly time
                try:
                    req = request_queue.get_nowait()
                    pending_batch.append(req)
                except queue.Empty:
                    break

            # Process current batch (GPU inference)
            current_batch = pending_batch
            pending_batch = []

            self._process_batch_requests(current_batch, response_queues)

            # After GPU finishes, immediately collect anything that arrived
            while True:
                try:
                    req = request_queue.get_nowait()
                    pending_batch.append(req)
                except queue.Empty:
                    break

        if self.verbose:
            print(f"BatchInferenceServer stopping. Stats: {self.total_batches} mega-batches, {self.total_evals} evals")

    def _process_batch_requests(self, batch_requests: list[BatchEvalRequest], response_queues: dict[int, mp.Queue]):
        """Merge batches, run inference, split results."""
        t0 = time.perf_counter()

        # Merge all requests into mega-batch
        all_states = []
        all_legal_actions = []
        request_sizes = []  # Track size of each request for splitting

        game_name = batch_requests[0].game_name
        game = game_registry.create_game(game_name)

        for req in batch_requests:
            request_sizes.append(len(req.action_histories))
            for action_history, legal_actions in zip(req.action_histories, req.legal_actions_list):
                all_states.append(_ActionHistoryState(action_history))
                all_legal_actions.append(legal_actions)

        # Run mega-batch
        results = self.evaluator.evaluate_batch(game, all_states, all_legal_actions)

        # Split results back to workers
        idx = 0
        for req, size in zip(batch_requests, request_sizes):
            policies = [results[idx + i].legal_policy for i in range(size)]
            values = [results[idx + i].player_values for i in range(size)]

            response = BatchEvalResponse(
                batch_id=req.batch_id,
                legal_policies=policies,
                player_values_list=values,
            )
            response_queues[req.worker_id].put(response)
            idx += size

        elapsed = time.perf_counter() - t0
        self.total_batches += 1
        self.total_evals += len(all_states)
        self.total_time += elapsed

        if self.verbose and self.total_batches % 50 == 0:
            mean_batch = self.total_evals / self.total_batches
            evals_per_sec = self.total_evals / self.total_time
            print(
                f"BatchInferenceServer: {self.total_batches} mega-batches, mean_size={mean_batch:.1f}, evals/sec={evals_per_sec:.1f}"
            )


class IPCBatchEvaluator:
    """
    Async evaluator that batches locally and sends to inference server via IPC.

    Similar to AsyncNetworkEvaluator but sends batches via multiprocessing queue.
    """

    def __init__(
        self,
        worker_id: int,
        game_name: str,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        max_batch_size: int = 128,
        max_wait_ms: float = 1.0,
    ):
        self.worker_id = worker_id
        self.game_name = game_name
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self.next_batch_id = 0
        self._pending: dict[int, list] = {}  # batch_id -> list of futures
        self._local_queue: asyncio.Queue = None
        self._worker_task = None
        self._stopping = False

    async def start(self):
        """Start the background worker."""
        self._local_queue = asyncio.Queue()
        self._stopping = False
        self._worker_task = asyncio.create_task(self._worker_run())

    async def stop(self):
        """Stop the background worker."""
        self._stopping = True
        if self._worker_task:
            await self._local_queue.put(None)  # Sentinel
            await self._worker_task

    async def _worker_run(self):
        """Background worker that batches and sends to server."""
        while not self._stopping:
            batch = await self._collect_local_batch()
            if batch:
                await self._send_batch(batch)

    async def _collect_local_batch(self) -> list:
        """Collect requests from local async queue."""
        batch = []

        # Wait for first item
        item = await self._local_queue.get()
        if item is None:
            return []
        batch.append(item)

        # Collect more until batch full or queue empty
        deadline = asyncio.get_event_loop().time() + self.max_wait_ms / 1000
        while len(batch) < self.max_batch_size:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self._local_queue.get(), timeout=remaining)
                if item is None:
                    break
                batch.append(item)
            except asyncio.TimeoutError:
                break

        return batch

    async def _send_batch(self, batch: list):
        """Send batch to server and handle response."""
        batch_id = self.next_batch_id
        self.next_batch_id += 1

        # Prepare batch request
        action_histories = [item[0] for item in batch]
        legal_actions_list = [item[1] for item in batch]
        futures = [item[2] for item in batch]

        self._pending[batch_id] = futures

        req = BatchEvalRequest(
            worker_id=self.worker_id,
            batch_id=batch_id,
            game_name=self.game_name,
            action_histories=action_histories,
            legal_actions_list=legal_actions_list,
        )

        # Send to server (blocking put in thread to not block event loop)
        await asyncio.get_event_loop().run_in_executor(None, self.request_queue.put, req)

        # Wait for response (blocking get in thread)
        response = await asyncio.get_event_loop().run_in_executor(None, self.response_queue.get)

        # Distribute results
        if response.batch_id in self._pending:
            futures = self._pending.pop(response.batch_id)
            for future, policy, value in zip(futures, response.legal_policies, response.player_values_list):
                if not future.done():
                    future.set_result(
                        NetworkEvaluatorResult(
                            legal_policy=policy,
                            player_values=value,
                        )
                    )

    async def evaluate_async(self, game, state, legal_actions) -> NetworkEvaluatorResult:
        """Queue request and wait for result."""
        future = asyncio.get_event_loop().create_future()
        await self._local_queue.put((list(state.action_history), list(legal_actions), future))
        return await future


async def async_worker_process_main(
    worker_id: int,
    game_name: str,
    num_games: int,
    num_simulations: int,
    concurrent_games: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    seed: int,
):
    """Async main function for worker process."""
    from rgi.rgizero.players.alphazero import AlphazeroPlayer, play_game_async

    game = game_registry.create_game(game_name)
    rng = np.random.default_rng(seed)

    # Create IPC batch evaluator
    evaluator = IPCBatchEvaluator(
        worker_id=worker_id,
        game_name=game_name,
        request_queue=request_queue,
        response_queue=response_queue,
        max_batch_size=concurrent_games,
    )
    await evaluator.start()

    try:
        all_trajectories = []
        games_played = 0

        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_games)

        async def play_one():
            nonlocal games_played
            async with semaphore:
                player = AlphazeroPlayer(
                    game=game,
                    evaluator=evaluator,
                    rng=np.random.default_rng(rng.integers(0, 2**31)),
                    add_noise=True,
                    simulations=num_simulations,
                )
                result = await play_game_async(game, [player, player])
                games_played += 1
                if games_played % 10 == 0:
                    print(f"Worker {worker_id}: {games_played}/{num_games} games")
                return result

        # Play all games
        tasks = [play_one() for _ in range(num_games)]
        all_trajectories = await asyncio.gather(*tasks)

    finally:
        await evaluator.stop()

    result_queue.put((worker_id, list(all_trajectories)))


def async_worker_process(
    worker_id: int,
    game_name: str,
    num_games: int,
    num_simulations: int,
    concurrent_games: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    seed: int,
):
    """Worker process entry point - runs async event loop."""
    import asyncio

    asyncio.run(
        async_worker_process_main(
            worker_id,
            game_name,
            num_games,
            num_simulations,
            concurrent_games,
            request_queue,
            response_queue,
            result_queue,
            seed,
        )
    )


class HybridParallelSelfPlay:
    """
    Hybrid parallel self-play: async workers + batch IPC + GPU server.

    Each worker runs 50-100 concurrent async games, batches locally,
    sends batches to GPU server for inference.
    """

    def __init__(
        self,
        model: ActionHistoryTransformer,
        game_name: str,
        vocab: Vocab,
        device: str,
        num_workers: int = 4,
        concurrent_games_per_worker: int = 50,
        num_simulations: int = 50,
        verbose: bool = False,
    ):
        self.model = model
        self.game_name = game_name
        self.vocab = vocab
        self.device = device
        self.num_workers = num_workers
        self.concurrent_games_per_worker = concurrent_games_per_worker
        self.num_simulations = num_simulations
        self.verbose = verbose

    def play_games(self, total_games: int, seed: int = 42) -> list:
        """Play games using hybrid parallel approach."""
        games_per_worker = total_games // self.num_workers
        remainder = total_games % self.num_workers

        # Queues
        request_queue = mp.Queue()
        response_queues = {i: mp.Queue() for i in range(self.num_workers)}
        result_queue = mp.Queue()
        stop_event = mp.Event()

        # Inference server
        server = BatchInferenceServer(
            model=self.model,
            vocab=self.vocab,
            device=self.device,
            verbose=self.verbose,
        )

        # Start workers
        workers = []
        rng = np.random.default_rng(seed)

        for i in range(self.num_workers):
            num_games = games_per_worker + (1 if i < remainder else 0)
            worker_seed = int(rng.integers(0, 2**31))

            p = mp.Process(
                target=async_worker_process,
                args=(
                    i,
                    self.game_name,
                    num_games,
                    self.num_simulations,
                    self.concurrent_games_per_worker,
                    request_queue,
                    response_queues[i],
                    result_queue,
                    worker_seed,
                ),
            )
            workers.append(p)
            p.start()

        # Run server in thread
        import threading

        server_thread = threading.Thread(
            target=server.run,
            args=(request_queue, response_queues, stop_event),
        )
        server_thread.start()

        # Collect results
        all_trajectories = []
        for _ in range(self.num_workers):
            worker_id, trajectories = result_queue.get()
            all_trajectories.extend(trajectories)
            if self.verbose:
                print(f"Collected {len(trajectories)} trajectories from worker {worker_id}")

        stop_event.set()
        server_thread.join(timeout=5.0)
        for p in workers:
            p.join(timeout=5.0)

        return all_trajectories
