#!/usr/bin/env python3
"""
gRPC benchmark using grpc.aio (native async client).

True async - no threads, pure asyncio integration.
"""

import sys
import time
import asyncio
import multiprocessing as mp
from pathlib import Path
from concurrent import futures
import queue
import threading

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import grpc
import grpc.aio
import numpy as np
import torch

import inference_pb2
import inference_pb2_grpc

from rgi.rgizero.common import TOKENS
from rgi.rgizero.data.trajectory_dataset import Vocab
from rgi.rgizero.games import game_registry
from rgi.rgizero.models.transformer import TransformerConfig
from rgi.rgizero.models.tuner import create_random_model
from rgi.rgizero.evaluators import ActionHistoryTransformerEvaluator
from rgi.rgizero.players.alphazero import AlphazeroPlayer, play_game_async, NetworkEvaluatorResult


class BatchCombiningServicer(inference_pb2_grpc.InferenceServiceServicer):
    """Same server as before - combines batches from workers."""
    
    def __init__(self, vocab_size: int, num_players: int, verbose: bool = True):
        self.verbose = verbose
        self.vocab_size = vocab_size
        self.num_players = num_players
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        game = game_registry.create_game("othello")
        vocab = Vocab(itos=[TOKENS.START_OF_GAME] + list(game.base_game.all_actions()))
        
        model_config = TransformerConfig(
            n_max_context=100, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=False,
        )
        model = create_random_model(model_config, vocab.vocab_size, num_players, seed=42, device=device)
        
        self.evaluator = ActionHistoryTransformerEvaluator(
            model=model, device=device, block_size=100, vocab=vocab, verbose=False,
        )
        
        self.request_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        self.total_batches = 0
        self.total_evals = 0
        self.start_time = time.time()
        
        self.batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self.batch_thread.start()
        
        if verbose:
            print(f"gRPC server ready on {device}")
    
    def _batch_processor(self):
        while not self.stop_event.is_set():
            pending = []
            
            try:
                req = self.request_queue.get(timeout=0.01)
                pending.append(req)
            except queue.Empty:
                continue
            
            while len(pending) < 100:
                try:
                    req = self.request_queue.get_nowait()
                    pending.append(req)
                except queue.Empty:
                    break
            
            total_B = sum(r[0].batch_size for r in pending)
            max_len = max(r[0].max_len for r in pending)
            
            x_combined = np.zeros((total_B, max_len), dtype=np.int32)
            len_combined = np.zeros(total_B, dtype=np.int32)
            mask_combined = np.zeros((total_B, self.vocab_size), dtype=np.bool_)
            
            idx = 0
            bounds = []
            for request, event, response_holder in pending:
                B = request.batch_size
                req_max_len = request.max_len
                
                x_req = np.array(request.x_data, dtype=np.int32).reshape(B, req_max_len)
                x_combined[idx:idx+B, :req_max_len] = x_req
                len_combined[idx:idx+B] = request.encoded_lengths
                
                mask_req = np.array(request.legal_mask, dtype=np.bool_).reshape(B, self.vocab_size)
                mask_combined[idx:idx+B] = mask_req
                
                bounds.append((idx, B, request, event, response_holder))
                idx += B
            
            legal_indices = [np.where(mask_combined[i])[0] for i in range(total_B)]
            results = self.evaluator.infer_from_encoded(
                x_combined, len_combined, mask_combined, legal_indices
            )
            
            for start_idx, count, request, event, response_holder in bounds:
                worker_results = results[start_idx:start_idx+count]
                
                response = inference_pb2.EncodedEvalResponse(
                    worker_id=request.worker_id,
                    request_id=request.request_id,
                    num_players=self.num_players,
                )
                
                for result in worker_results:
                    response.legal_policies.extend(result.legal_policy.tolist())
                    response.player_values.extend(result.player_values.tolist())
                
                response_holder['response'] = response
                event.set()
            
            self.total_batches += 1
            self.total_evals += total_B
            
            if self.verbose and self.total_batches % 50 == 0:
                elapsed = time.time() - self.start_time
                print(f"Server: batches={self.total_batches}, evals={self.total_evals}, "
                      f"combined={total_B}, evals/sec={self.total_evals/elapsed:.0f}")
    
    def EvaluateEncoded(self, request, context):
        event = threading.Event()
        response_holder = {}
        self.request_queue.put((request, event, response_holder))
        event.wait(timeout=30)
        return response_holder.get('response', inference_pb2.EncodedEvalResponse())
    
    def stop(self):
        self.stop_event.set()
        self.batch_thread.join(timeout=2)


def run_server(vocab_size: int, num_players: int, port: int, 
               ready_event: mp.Event, stop_event: mp.Event):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=50))
    servicer = BatchCombiningServicer(vocab_size, num_players, verbose=True)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    ready_event.set()
    
    while not stop_event.is_set():
        time.sleep(0.1)
    
    servicer.stop()
    server.stop(grace=1)
    
    elapsed = time.time() - servicer.start_time
    print(f"\nServer done: {servicer.total_batches} batches, {servicer.total_evals} evals in {elapsed:.1f}s")


def worker_process(worker_id: int, game_name: str, vocab_tokens: list,
                   num_games: int, concurrent_games: int, num_simulations: int,
                   port: int, result_queue: mp.Queue):
    """Worker using grpc.aio - native async client."""
    
    game = game_registry.create_game(game_name)
    vocab = Vocab(itos=vocab_tokens)
    vocab_size = len(vocab.stoi)
    
    async def run_games():
        # Initialize async gRPC
        channel = grpc.aio.insecure_channel(f'localhost:{port}')
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        
        rng = np.random.default_rng(42 + worker_id)
        semaphore = asyncio.Semaphore(concurrent_games)
        
        class AsyncGrpcEvaluator:
            """Evaluator using native async gRPC."""
            
            def __init__(self):
                self.batch_queue = asyncio.Queue()
                self.request_id = 0
                self.batch_task = None
                self.batch_count = 0
            
            async def start(self):
                self.batch_task = asyncio.create_task(self._batch_loop())
            
            async def stop(self):
                if self.batch_task:
                    self.batch_task.cancel()
            
            async def evaluate_async(self, game, state, legal_actions):
                future = asyncio.get_event_loop().create_future()
                await self.batch_queue.put((state, legal_actions, future))
                return await future
            
            async def _batch_loop(self):
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
                    max_len = max(len(r) for r in encoded_rows)
                    
                    x_np = np.zeros((B, max_len), dtype=np.int32)
                    for i, row in enumerate(encoded_rows):
                        x_np[i, :len(row)] = row
                    
                    encoded_len_np = np.array([len(r) for r in encoded_rows], dtype=np.int32)
                    
                    legal_mask = np.zeros((B, vocab_size), dtype=np.bool_)
                    num_legal_actions = []
                    for i, legal_actions in enumerate(legal_actions_list):
                        indices = [vocab.stoi[a] for a in legal_actions]
                        legal_mask[i, indices] = True
                        num_legal_actions.append(len(legal_actions))
                    
                    # Build request
                    request = inference_pb2.EncodedEvalRequest(
                        worker_id=worker_id,
                        request_id=self.request_id,
                        batch_size=B,
                        max_len=max_len,
                        vocab_size=vocab_size,
                    )
                    self.request_id += 1
                    
                    request.x_data.extend(x_np.flatten().tolist())
                    request.encoded_lengths.extend(encoded_len_np.tolist())
                    request.legal_mask.extend(legal_mask.flatten().tolist())
                    request.num_legal_actions.extend(num_legal_actions)
                    
                    self.batch_count += 1
                    if self.batch_count % 100 == 0:
                        print(f"Worker {worker_id}: sent async batch {self.batch_count} of {B}")
                    
                    # Async gRPC call - yields control!
                    response = await stub.EvaluateEncoded(request)
                    
                    # Set results
                    num_players = response.num_players
                    policy_offset = 0
                    value_offset = 0
                    
                    for i, future in enumerate(futures):
                        n_legal = num_legal_actions[i]
                        legal_policy = np.array(response.legal_policies[policy_offset:policy_offset+n_legal])
                        player_values = np.array(response.player_values[value_offset:value_offset+num_players])
                        
                        policy_offset += n_legal
                        value_offset += num_players
                        
                        if not future.done():
                            future.set_result(NetworkEvaluatorResult(
                                legal_policy=legal_policy,
                                player_values=player_values,
                            ))
        
        evaluator = AsyncGrpcEvaluator()
        await evaluator.start()
        
        completed = 0
        
        async def play_one():
            nonlocal completed
            async with semaphore:
                player = AlphazeroPlayer(
                    game=game,
                    evaluator=evaluator,
                    rng=np.random.default_rng(rng.integers(0, 2**31)),
                    add_noise=True,
                    simulations=num_simulations,
                )
                result = await play_game_async(game, [player, player])
                completed += 1
                if completed % 50 == 0:
                    print(f"Worker {worker_id}: {completed}/{num_games} games")
                return result
        
        tasks = [play_one() for _ in range(num_games)]
        results = await asyncio.gather(*tasks)
        
        await evaluator.stop()
        await channel.close()
        return results
    
    results = asyncio.run(run_games())
    result_queue.put((worker_id, len(results)))


def main():
    print("=" * 60)
    print("gRPC BENCHMARK (grpc.aio - Native Async)")
    print("=" * 60)
    
    game_name = "othello"
    total_games = 5000
    num_workers = 11
    games_per_worker = (total_games // num_workers) + 1
    concurrent_per_worker = 500
    num_simulations = 30
    port = 50051
    
    print(f"Workers: {num_workers}")
    print(f"Games/worker: {games_per_worker}")
    print(f"Concurrent/worker: {concurrent_per_worker}")
    print(f"Total concurrent: {num_workers * concurrent_per_worker}")
    
    game = game_registry.create_game(game_name)
    vocab_tokens = [TOKENS.START_OF_GAME] + list(game.base_game.all_actions())
    vocab_size = len(vocab_tokens)
    num_players = game.num_players(game.initial_state())
    
    ready_event = mp.Event()
    stop_event = mp.Event()
    
    server_proc = mp.Process(
        target=run_server,
        args=(vocab_size, num_players, port, ready_event, stop_event)
    )
    server_proc.start()
    
    ready_event.wait(timeout=10)
    time.sleep(0.5)
    
    print("\nStarting workers...")
    start_time = time.time()
    
    result_queue = mp.Queue()
    workers = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(i, game_name, vocab_tokens, games_per_worker, concurrent_per_worker,
                  num_simulations, port, result_queue)
        )
        workers.append(p)
        p.start()
    
    total_games = 0
    for _ in range(num_workers):
        try:
            w_id, n_games = result_queue.get(timeout=300)
            total_games += n_games
            print(f"Worker {w_id} done")
        except:
            print("Worker timeout")
    
    duration = time.time() - start_time
    
    stop_event.set()
    server_proc.join(timeout=5)
    
    for p in workers:
        p.join(timeout=2)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Games: {total_games}")
    print(f"Duration: {duration:.2f}s")
    print(f"Games/sec: {total_games / duration:.2f}")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
