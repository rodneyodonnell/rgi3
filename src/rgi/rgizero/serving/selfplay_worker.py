"""
Multi-process worker for gRPC-based self-play.

Each worker runs in a separate process with its own gRPC client,
playing games and returning results via a multiprocessing Queue.
"""

import asyncio
import multiprocessing as mp

import numpy as np
import grpc

from rgi.rgizero.games import game_registry
from rgi.rgizero.data.trajectory_dataset import Vocab
from rgi.rgizero.players.alphazero import AlphazeroPlayer, play_game_async


def run_selfplay_worker(
    worker_id: int,
    game_name: str,
    vocab_tokens: list,
    num_games: int,
    concurrent_games: int,
    num_simulations: int,
    port: int,
    seed: int,
    result_queue: mp.Queue,
):
    """Worker process that plays games using gRPC inference server.
    
    Args:
        worker_id: Unique ID for this worker
        game_name: Name of game to play
        vocab_tokens: Token list for vocab
        num_games: Number of games this worker should play
        concurrent_games: Max concurrent games per worker
        num_simulations: MCTS simulations per move
        port: gRPC server port
        seed: Random seed for this worker
        result_queue: Queue to put results
    """
    # Import proto stubs - must be done in worker process
    import sys
    from pathlib import Path
    # Add serving module to path if needed
    serving_path = Path(__file__).parent
    if str(serving_path) not in sys.path:
        sys.path.insert(0, str(serving_path))
    
    from . import inference_pb2, inference_pb2_grpc
    
    game = game_registry.create_game(game_name)
    vocab = Vocab(itos=vocab_tokens)
    vocab_size = len(vocab.stoi)
    
    async def run_games():
        # Initialize async gRPC
        channel = grpc.aio.insecure_channel(f'localhost:{port}')
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        
        rng = np.random.default_rng(seed)
        semaphore = asyncio.Semaphore(concurrent_games)
        
        class AsyncGrpcEvaluator:
            """Evaluator using native async gRPC with batching."""
            
            def __init__(self):
                self.batch_queue = asyncio.Queue()
                self.request_id = 0
                self.batch_task = None
            
            async def start(self):
                self.batch_task = asyncio.create_task(self._batch_loop())
            
            async def stop(self):
                if self.batch_task:
                    self.batch_task.cancel()
                    try:
                        await self.batch_task
                    except asyncio.CancelledError:
                        pass
            
            async def evaluate_async(self, game, state, legal_actions):
                future = asyncio.get_event_loop().create_future()
                await self.batch_queue.put((state, legal_actions, future))
                return await future
            
            async def _batch_loop(self):
                from rgi.rgizero.players.alphazero import NetworkEvaluatorResult
                
                while True:
                    batch = []
                    
                    # Wait for first
                    item = await self.batch_queue.get()
                    batch.append(item)
                    
                    # Grab all available immediately
                    while len(batch) < 2000:
                        try:
                            item = self.batch_queue.get_nowait()
                            batch.append(item)
                        except asyncio.QueueEmpty:
                            break
                    
                    states = [item[0] for item in batch]
                    legal_actions_list = [item[1] for item in batch]
                    futures = [item[2] for item in batch]
                    
                    # Encode
                    B = len(batch)
                    encoded_rows = [vocab.encode(s.action_history) for s in states]
                    lengths = [len(row) for row in encoded_rows]
                    max_len = max(lengths) if lengths else 1
                    
                    # Build padded arrays
                    x_np = np.zeros((B, max_len), dtype=np.int32)
                    for i, row in enumerate(encoded_rows):
                        x_np[i, :len(row)] = row
                    
                    encoded_len_np = np.array(lengths, dtype=np.int32)
                    
                    # Build legal mask
                    legal_mask = np.zeros((B, vocab_size), dtype=np.bool_)
                    num_legal_actions = []
                    for i, actions in enumerate(legal_actions_list):
                        encoded_actions = np.array(vocab.encode(actions), dtype=np.int64)
                        legal_mask[i, encoded_actions] = True
                        num_legal_actions.append(len(actions))
                    
                    num_legal_np = np.array(num_legal_actions, dtype=np.int32)
                    
                    # Build request with bytes
                    request = inference_pb2.EncodedEvalRequest(
                        worker_id=worker_id,
                        request_id=self.request_id,
                        batch_size=B,
                        max_len=max_len,
                        vocab_size=vocab_size,
                        x_data_bytes=x_np.tobytes(),
                        encoded_lengths_bytes=encoded_len_np.tobytes(),
                        legal_mask_bytes=legal_mask.tobytes(),
                        num_legal_actions_bytes=num_legal_np.tobytes(),
                    )
                    self.request_id += 1
                    
                    # Make gRPC call
                    response = await stub.EvaluateEncoded(request)
                    
                    # Parse response with zero-copy
                    all_policies = np.frombuffer(response.legal_policies_bytes, dtype=np.float32)
                    all_values = np.frombuffer(response.player_values_bytes, dtype=np.float32)
                    num_players = response.num_players
                    
                    # Distribute results
                    policy_offset = 0
                    value_offset = 0
                    for i, (state, legal_actions, future) in enumerate(zip(states, legal_actions_list, futures)):
                        n_legal = len(legal_actions)
                        policy = all_policies[policy_offset:policy_offset + n_legal]
                        values = all_values[value_offset:value_offset + num_players]
                        
                        result = NetworkEvaluatorResult(
                            legal_policy=policy,
                            player_values=values,
                        )
                        if not future.done():
                            future.set_result(result)
                        
                        policy_offset += n_legal
                        value_offset += num_players
        
        evaluator = AsyncGrpcEvaluator()
        await evaluator.start()
        
        async def play_one():
            player_seed = rng.integers(0, 2**31)
            player_rng = np.random.default_rng(player_seed)
            player = AlphazeroPlayer(
                game, evaluator, rng=player_rng,
                add_noise=True, simulations=num_simulations
            )
            async with semaphore:
                return await play_game_async(game, [player, player])
        
        tasks = [play_one() for _ in range(num_games)]
        results = await asyncio.gather(*tasks)
        
        await evaluator.stop()
        await channel.close()
        
        return results
    
    # Run the async games
    results = asyncio.run(run_games())
    result_queue.put((worker_id, results))
