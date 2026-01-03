"""
Evaluators for bridging Models and MCTS Players.
"""

from typing import override, Optional, Any
import time
import threading
import queue
from concurrent.futures import Future
from dataclasses import dataclass
import asyncio
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from rgi.rgizero.data.trajectory_dataset import Vocab
from rgi.rgizero.players.alphazero import NetworkEvaluator, NetworkEvaluatorResult
from rgi.rgizero.models.action_history_transformer import ActionHistoryTransformer


class ActionHistoryTransformerEvaluator(NetworkEvaluator):
    """Neural network evaluator for MCTS."""

    def __init__(self, model: ActionHistoryTransformer, device: str, block_size: int, vocab: Vocab, verbose=True):
        self.model = model.eval()
        self.device = device
        self.block_size = block_size
        self.vocab = vocab
        self.verbose = verbose
        self.total_time = 0.0
        self.total_evals = 0
        self.total_batches = 0

    @override
    @torch.no_grad()
    def evaluate(self, game, state, legal_actions) -> NetworkEvaluatorResult:
        return self.evaluate_batch(game, [state], [legal_actions])[0]

    def _maybe_pin(self, tensor):
        """Pin memory if on GPU."""
        # Only pin on GPU. Pinning on CPU raises an error about CUDA drivers not being available.
        if self.device == "cuda":
            return tensor.pin_memory()
        return tensor

    @override
    @torch.inference_mode()
    def evaluate_batch(self, game, states_list, legal_actions_list):
        """Full evaluation: encode states and run inference."""
        encoded = self.encode_batch(states_list, legal_actions_list)
        return self.infer_from_encoded(*encoded)
    
    def encode_batch(self, states_list, legal_actions_list) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
        """
        Encode states for inference. Can be called on worker side.
        
        Returns:
            x_np: Encoded action histories (B, max_len) int32
            encoded_len: Length of each encoded history (B,) int32
            legal_mask: Legal action mask (B, vocab_size) bool
            legal_indices: List of legal action indices per state
        """
        B = len(states_list)
        
        # Encode action histories
        encoded_rows = []
        encoded_len = []
        for state in states_list:
            encoded = self.vocab.encode(state.action_history)
            encoded_rows.append(encoded)
            encoded_len.append(len(encoded))
        max_encoded_len = max(encoded_len) if encoded_len else 1
        
        if max_encoded_len > self.block_size:
            raise ValueError(f"max_encoded_len {max_encoded_len} > block_size {self.block_size}")
        
        # Pad to numpy array
        x_np = np.zeros((B, max_encoded_len), dtype=np.int32)
        for i, row in enumerate(encoded_rows):
            x_np[i, :len(row)] = row
        
        encoded_len_np = np.array(encoded_len, dtype=np.int32)
        
        # Build legal action mask
        vocab_size = len(self.vocab.stoi)
        legal_mask = np.zeros((B, vocab_size), dtype=np.bool_)
        legal_indices = []
        for i, legal_actions in enumerate(legal_actions_list):
            indices = np.array([self.vocab.stoi[a] for a in legal_actions], dtype=np.int32)
            legal_mask[i, indices] = True
            legal_indices.append(indices)
        
        return x_np, encoded_len_np, legal_mask, legal_indices
    
    @torch.inference_mode()
    def infer_from_encoded(
        self,
        x_np: np.ndarray,
        encoded_len_np: np.ndarray,
        legal_mask: np.ndarray,
        legal_indices: list[np.ndarray],
    ) -> list[NetworkEvaluatorResult]:
        """
        Run inference from pre-encoded inputs. Runs on GPU.
        
        Args:
            x_np: Encoded action histories (B, max_len) int32
            encoded_len_np: Length of each encoded sequence (B,) int32
            legal_mask: Legal action mask (B, vocab_size) bool
            legal_indices: List of legal action indices per state
        
        Returns:
            List of NetworkEvaluatorResult
        """
        t0 = time.time()
        B = len(x_np)
        
        # Transfer to GPU
        x_pinned = self._maybe_pin(torch.from_numpy(x_np))
        x_gpu = x_pinned.to(self.device, non_blocking=True)
        
        encoded_len_pinned = self._maybe_pin(torch.from_numpy(encoded_len_np))
        encoded_len_gpu = encoded_len_pinned.to(self.device, non_blocking=True)
        
        # Run model
        (policy_logits_gpu, value_logits_gpu), _, _ = self.model(x_gpu, encoded_len=encoded_len_gpu)
        
        policy_logits_gpu = policy_logits_gpu.squeeze(1)
        value_logits_gpu = value_logits_gpu.squeeze(1)
        
        # Apply legal mask
        legal_policy_mask_pinned = self._maybe_pin(torch.from_numpy(legal_mask))
        legal_policy_mask_gpu = legal_policy_mask_pinned.to(self.device, non_blocking=True)
        masked_policy_logits_gpu = policy_logits_gpu.masked_fill(~legal_policy_mask_gpu, float("-inf"))
        
        policy = F.softmax(masked_policy_logits_gpu, dim=-1)
        val_probs = F.softmax(value_logits_gpu, dim=-1)
        value = val_probs * 2 - 1  # map to [-1, 1]
        
        # Transfer back to CPU
        policy_np_batch = policy.cpu().numpy()
        value_np_batch = value.cpu().numpy()
        
        # Extract results
        ret = []
        for i, (policy_np, value_np, mask) in enumerate(zip(policy_np_batch, value_np_batch, legal_mask)):
            legal_policy = policy_np[mask]
            ret.append(NetworkEvaluatorResult(legal_policy, value_np))
        
        t1 = time.time()
        self.total_time += t1 - t0
        self.total_evals += B
        self.total_batches += 1
        if self.verbose and self.total_batches % 1000 == 0:
            print(
                f"Evaluation time: {t1 - t0:.3f} seconds, size={B}, eval-per-second={B / (t1 - t0):.2f}, total-batches={self.total_batches}, mean-eval-per-second={self.total_evals / self.total_time:.2f}, mean-time-per-batch={self.total_time / self.total_batches:.3f}, mean-batch-size={self.total_evals / self.total_batches:.2f}"
            )
        return ret


@dataclass
class EvalReq:
    game: Any
    state: Any
    legal_actions: list[Any]
    future: Future


class QueuedNetworkEvaluator(NetworkEvaluator):
    def __init__(
        self,
        base_evaluator: ActionHistoryTransformerEvaluator,
        max_batch_size=1024,
        max_latency_ms=1,
        auto_start=True,
        verbose=False,
    ):
        self.evaluator = base_evaluator
        self.max_batch_size = max_batch_size
        self.max_latency_ms = max_latency_ms
        self.verbose = verbose
        self.queue: queue.Queue[EvalReq] = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        if auto_start:
            self.start()

    @override
    def evaluate(self, game, state: Any, legal_actions: list[Any]) -> NetworkEvaluatorResult:
        future = Future()
        self.queue.put(EvalReq(game, state, legal_actions, future))
        return future.result()

    def start(self):
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def run(self):
        while not self._stop.is_set():
            batch = self._collect()
            if batch:
                self._run_once(batch)

    def _collect(self) -> list[EvalReq]:
        batch = []
        start = time.perf_counter()

        # block for first item
        item = self.queue.get()
        batch.append(item)

        # Fill batch to capacity.
        # Only stop when 'batch is full', or 'queue is empty AND timeout elapsed'
        while len(batch) < self.max_batch_size:
            remain = self.max_latency_ms / 1000 - (time.perf_counter() - start)
            try:
                batch.append(self.queue.get(timeout=max(0, remain)))
            except queue.Empty:
                break
        return batch

    def _run_once(self, batch: list[EvalReq]):
        if self.verbose:
            print(f"QueuedNetworkEvaluator._run_once, batch_size={len(batch)}")
        states = [r.state for r in batch]
        legal = [r.legal_actions for r in batch]
        game = batch[0].game
        try:
            outs = self.evaluator.evaluate_batch(game, states, legal)
            for r, out in zip(batch, outs):
                r.future.set_result(out)
        except Exception as e:
            for r in batch:
                r.future.set_exception(e)


@dataclass
class AsyncEvalReq:
    game: Any
    state: Any
    legal_actions: list[Any]
    future: asyncio.Future


class AsyncNetworkEvaluator(NetworkEvaluator):
    def __init__(
        self,
        base_evaluator: NetworkEvaluator,
        max_batch_size: int = 1024,
        verbose=False,
    ):
        self.evaluator = base_evaluator
        self.max_batch_size = max_batch_size
        self.queue: asyncio.Queue[AsyncEvalReq] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._stopping = False
        self.verbose = verbose

    async def start(self):
        if self._worker_task is None or self._worker_task.done():
            self._stopping = False
            self._worker_task = asyncio.create_task(self._worker_run())

    async def stop(self):
        self._stopping = True
        if self._worker_task:
            await self.queue.put(AsyncEvalReq(None, None, [], asyncio.Future()))  # Sentinel to wake up worker
            await self._worker_task
            self._worker_task = None

    async def _worker_run(self):
        batch_deque: deque[AsyncEvalReq] = deque()
        while not self._stopping:
            # Block waiting for first item.
            item = await self.queue.get()
            if item.state is None and item.legal_actions == []:  # Sentinel check
                break
            batch_deque.append(item)

            # Collect more items until max_batch_size or queue empty
            while len(batch_deque) < self.max_batch_size:
                try:
                    item = self.queue.get_nowait()
                    if item.state is None and item.legal_actions == []:  # Sentinel check
                        self.queue.put_nowait(item)  # Put back for proper stop
                        break
                    batch_deque.append(item)
                except asyncio.QueueEmpty:
                    break

            if batch_deque:  # Process the batch if not empty
                batch_list = list(batch_deque)
                batch_deque.clear()
                await asyncio.to_thread(self._run_once, batch_list)

    def _run_once(self, batch: list[AsyncEvalReq]):
        states = [r.state for r in batch]
        legal_actions = [r.legal_actions for r in batch]
        game = batch[0].game
        if self.verbose:
            print(f"batch size: {len(batch)}")

        try:
            outs = self.evaluator.evaluate_batch(game, states, legal_actions)
            for req, out in zip(batch, outs):
                req.future.set_result(out)
        except Exception as e:
            for req in batch:
                req.future.set_exception(e)

    @override
    async def evaluate_async(self, game, state: Any, legal_actions: list[Any]) -> NetworkEvaluatorResult:
        if self._worker_task is None:
            raise RuntimeError(f"{self.__class__.__name__} worker not running. Missing `await evaluator.start()`?")
        future = asyncio.Future()
        await self.queue.put(AsyncEvalReq(game, state, legal_actions, future))
        result = await future
        return result
