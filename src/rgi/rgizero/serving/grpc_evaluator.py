"""
gRPC client evaluator for use in players.

Batches requests and sends to inference server via gRPC.
"""

import asyncio
from typing import Optional

import grpc.aio
import numpy as np

from rgi.rgizero.serving import inference_pb2, inference_pb2_grpc
from rgi.rgizero.players.alphazero import NetworkEvaluatorResult


class GrpcEvaluator:
    """Evaluator that sends requests to a gRPC inference server.
    
    Uses async batching - collects requests, sends to server, distributes results.
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        vocab,  # Vocab object for encoding
        vocab_size: int,
        worker_id: int = 0,
    ):
        self.host = host
        self.port = port
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.worker_id = worker_id
        
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub = None
        
        self.batch_queue: asyncio.Queue = asyncio.Queue()
        self.batch_task: Optional[asyncio.Task] = None
        self.request_id = 0
        self.batch_count = 0
    
    async def connect(self):
        """Connect to the inference server."""
        self.channel = grpc.aio.insecure_channel(f'{self.host}:{self.port}')
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)
        self.batch_task = asyncio.create_task(self._batch_loop())
    
    async def close(self):
        """Close the connection."""
        if self.batch_task:
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass
        if self.channel:
            await self.channel.close()
    
    async def evaluate_async(self, game, state, legal_actions) -> NetworkEvaluatorResult:
        """Called by players - puts request on queue, returns future."""
        future = asyncio.get_event_loop().create_future()
        await self.batch_queue.put((state, legal_actions, future))
        return await future
    
    async def _batch_loop(self):
        """Background task: collects batch, calls gRPC, sets results."""
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
            encoded_rows = [self.vocab.encode(s.action_history) for s in states]
            max_len = max(len(r) for r in encoded_rows)
            
            x_np = np.zeros((B, max_len), dtype=np.int32)
            for i, row in enumerate(encoded_rows):
                x_np[i, :len(row)] = row
            
            encoded_len_np = np.array([len(r) for r in encoded_rows], dtype=np.int32)
            
            legal_mask = np.zeros((B, self.vocab_size), dtype=np.bool_)
            num_legal_actions = []
            for i, legal_actions in enumerate(legal_actions_list):
                indices = [self.vocab.stoi[a] for a in legal_actions]
                legal_mask[i, indices] = True
                num_legal_actions.append(len(legal_actions))
            
            # Build request with bytes for zero-copy
            num_legal_np = np.array(num_legal_actions, dtype=np.int32)
            request = inference_pb2.EncodedEvalRequest(
                worker_id=self.worker_id,
                request_id=self.request_id,
                batch_size=B,
                max_len=max_len,
                vocab_size=self.vocab_size,
                x_data_bytes=x_np.tobytes(),
                encoded_lengths_bytes=encoded_len_np.tobytes(),
                legal_mask_bytes=legal_mask.tobytes(),
                num_legal_actions_bytes=num_legal_np.tobytes(),
            )
            self.request_id += 1
            self.batch_count += 1
            
            # Async gRPC call
            try:
                response = await self.stub.EvaluateEncoded(request)
            except Exception as e:
                # On error, fail all futures
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
                continue
            
            # Set results from bytes
            num_players = response.num_players
            all_policies = np.frombuffer(response.legal_policies_bytes, dtype=np.float32)
            all_values = np.frombuffer(response.player_values_bytes, dtype=np.float32)
            
            policy_offset = 0
            value_offset = 0
            
            for i, future in enumerate(futures):
                n_legal = num_legal_actions[i]
                legal_policy = all_policies[policy_offset:policy_offset+n_legal].copy()
                player_values = all_values[value_offset:value_offset+num_players].copy()
                
                policy_offset += n_legal
                value_offset += num_players
                
                if not future.done():
                    future.set_result(NetworkEvaluatorResult(
                        legal_policy=legal_policy,
                        player_values=player_values,
                    ))
