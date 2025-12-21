"""
Multitask transformer from action-history to Policy and Value heads.
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
import torch.nn as nn
import torch.nn.functional as F

from rgi.rgizero.models.transformer import TransformerConfig, Transformer, LayerNorm, init_weights
from rgi.rgizero.models.token_transformer import TokenTransformer
from rgi.rgizero.data.trajectory_dataset import Vocab
from rgi.rgizero.players.alphazero import NetworkEvaluator, NetworkEvaluatorResult
from rgi.rgizero.utils import validate_probabilities_or_die


# TODO: Move to util class.
def validate_probabilities_or_die(tensor: torch.Tensor, dim: int = 1, tol: float = 1e-6) -> bool:
    # 1. Check if all values are >= 0 and <= 1
    in_range = (tensor >= 0).all() and (tensor <= 1).all()
    if not in_range:
        raise ValueError(f"Probabilities are not in range [0, 1]: {tensor}")

    # 2. Check if sum is approximately 1.0
    row_sums = tensor.sum(dim=dim)
    sums_to_one = torch.allclose(row_sums, torch.ones_like(row_sums), atol=tol)
    if not sums_to_one:
        raise ValueError(f"Probabilities do not sum to 1.0: {tensor}, sums: {row_sums}")

    return bool(in_range) and bool(sums_to_one)


class PolicyValueHead(nn.Module):
    def __init__(self, n_embd: int, num_actions: int, num_players: int):
        super().__init__()
        self.policy = nn.Linear(n_embd, num_actions)  # p(action) logits for each action
        self.value = nn.Linear(n_embd, num_players)  # p(win) logits for each player

    def forward(self, last_hidden_layer):
        # last_hidden_layer: (B, n_embd)
        policy_logits = self.policy(last_hidden_layer)
        value_logits = self.value(last_hidden_layer)
        return policy_logits, value_logits


class ActionHistoryTransformer(nn.Module):
    def __init__(self, config: TransformerConfig, action_vocab_size: int, num_players: int):
        super().__init__()
        self.action_vocab_size = action_vocab_size
        self.num_players = num_players
        self.config = config
        self.action_embedding = nn.Embedding(action_vocab_size, config.n_embd)
        self.transformer = Transformer(config)
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.policy_value_head = PolicyValueHead(config.n_embd, action_vocab_size, num_players)
        self.policy_value_head.policy.weight = self.action_embedding.weight  # weight tying
        self.apply(init_weights)

    def forward(
        self,
        idx: torch.Tensor,
        policy_target: Optional[torch.Tensor] = None,  # (B, T, num_actions)
        value_target: Optional[torch.Tensor] = None,  # (B, T, num_players)
        padding_mask: Optional[torch.Tensor] = None,  # (B, T)
        encoded_len: Optional[torch.Tensor] = None,  # (B)
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor], torch.Tensor | float | None
    ]:  # ((policy_logits, value_logits), loss_dict, loss)
        """Forward pass for ActionHistoryTransformer.

        Args:
            idx: Action sequence tokens (B, T)
            policy_target: Target policy distributions (B, T, num_actions) - optional for training
            value_target: Target value distributions (B, T, num_players) - optional for training
            padding_mask: Mask of padding tokens, '1' for non-padding, '0' for padding (B, T) - optional for training

        Returns:
            ((policy_logits, value_logits), loss) where:
            - policy_logits: (B, T, num_actions) for training, (B, 1, num_actions) for inference
            - value_logits: (B, T, num_players) for training, (B, 1, num_players) for inference
            - loss: scalar loss for training, None for inference
        """
        B, T = idx.shape  # B = batch size, T = sequence length

        # Embed actions and pass through transformer
        action_emb = self.action_embedding(idx)  # (B, T, n_embd)
        h = self.transformer(action_emb)  # (B, T, n_embd)
        h = self.ln_f(h)  # (B, T, n_embd)

        loss_dict = {}

        if policy_target is not None or value_target is not None:
            policy_logits, value_logits = self.policy_value_head(h)
            flat_padding_mask = padding_mask.view(-1) if padding_mask is not None else None
            # Compute losses
            loss = 0.0
            if policy_target is not None:
                # Flatten for cross entropy: (B*T, num_actions) and (B*T, num_actions)
                flat_policy_logits = policy_logits.view(-1, self.action_vocab_size)
                flat_policy_target = policy_target.view(-1, self.action_vocab_size)
                # Mask out padding tokens so they are not considered in the loss or gradients.
                if flat_padding_mask is not None:
                    flat_policy_logits = flat_policy_logits[flat_padding_mask]
                    flat_policy_target = flat_policy_target[flat_padding_mask]
                validate_probabilities_or_die(flat_policy_target)
                # Calcualte the average loss per unpadded tokens.
                # note: We may want to experiment with average per batch?
                policy_loss = F.cross_entropy(flat_policy_logits, flat_policy_target, reduction="mean")
                loss_dict["policy_loss"] = policy_loss
                loss += policy_loss

            if value_target is not None:
                flat_value_logits = value_logits.view(-1, self.num_players)
                flat_value_target = value_target.view(-1, self.num_players)
                # Mask out padding tokens so they are not considered in the loss or gradients.
                if flat_padding_mask is not None:
                    flat_value_logits = flat_value_logits[flat_padding_mask]
                    flat_value_target = flat_value_target[flat_padding_mask]
                validate_probabilities_or_die(flat_value_target)
                value_loss = F.cross_entropy(flat_value_logits, flat_value_target, reduction="mean")
                loss_dict["value_loss"] = value_loss
                loss += value_loss
        else:
            # Inference mode: only compute logits for final position
            if encoded_len is not None:
                batch_idx = torch.arange(B, device=h.device)
                h_last = h[batch_idx, encoded_len - 1].unsqueeze(1)
            else:
                h_last = h[:, [-1], :]
            policy_logits, value_logits = self.policy_value_head(h_last)
            loss = None

        return (policy_logits, value_logits), loss_dict, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return TokenTransformer.configure_optimizers(self, weight_decay, learning_rate, betas, device_type)


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
        B = len(states_list)
        L = self.block_size

        t0 = time.time()
        # encode rows
        encoded_rows = []
        encoded_len = []
        max_encoded_len = 0
        for state in states_list:
            encoded = self.vocab.encode(state.action_history)
            encoded_rows.append(encoded)
            encoded_len.append(len(encoded))
        max_encoded_len = max(encoded_len)

        if max_encoded_len > L:
            raise ValueError(f"max_encoded_len {max_encoded_len} > block_size {L}")

        # Start with zeros to simplify padding.
        x_np = np.zeros((B, max_encoded_len), dtype=np.int32)
        for i, encoded_row in enumerate(encoded_rows):
            x_np[i, : len(encoded_row)] = encoded_row
        x_pinned = self._maybe_pin(torch.from_numpy(x_np))

        # Process model on GPU.
        x_gpu = x_pinned.to(self.device, non_blocking=True)
        encoded_len_pinned = self._maybe_pin(torch.tensor(encoded_len))
        encoded_len_gpu = encoded_len_pinned.to(self.device, non_blocking=True)
        (policy_logits_gpu, value_logits_gpu), _, _ = self.model(x_gpu, encoded_len=encoded_len_gpu)

        policy_logits_gpu = policy_logits_gpu.squeeze(1)
        value_logits_gpu = value_logits_gpu.squeeze(1)

        # Calculate legal_policy_mask on CPU
        legal_policy_mask_np = np.zeros(policy_logits_gpu.shape, dtype=np.bool_)
        for i, legal_actions in enumerate(legal_actions_list):
            legal_idx = np.array([self.vocab.stoi[a] for a in legal_actions], dtype=np.int32)
            legal_policy_mask_np[i, legal_idx] = True
        legal_policy_mask_pinned = self._maybe_pin(torch.from_numpy(legal_policy_mask_np))

        # Process masked softmax on GPU
        legal_policy_mask_gpu = legal_policy_mask_pinned.to(self.device, non_blocking=True)
        masked_policy_logits_gpu = policy_logits_gpu.masked_fill(~legal_policy_mask_gpu, float("-inf"))

        policy = F.softmax(masked_policy_logits_gpu, dim=-1)  # [B, V]
        val_probs = F.softmax(value_logits_gpu, dim=-1)  # [B, 3]
        value = val_probs * 2 - 1  # map to [-1, 1]

        # Blocks waiting for GPU to complete.
        policy_np_batch = policy.cpu().numpy()
        value_np_batch = value.cpu().numpy()

        ret = [None] * len(states_list)
        for i, (policy_np, value_np, mask) in enumerate(zip(policy_np_batch, value_np_batch, legal_policy_mask_np)):
            legal_policy = policy_np[mask]
            ret[i] = NetworkEvaluatorResult(legal_policy, value_np)
        t1 = time.time()
        self.total_time += t1 - t0
        self.total_evals += len(states_list)
        self.total_batches += 1
        if self.verbose and self.total_batches % 1000 == 0:
            print(
                f"Evaluation time: {t1 - t0:.3f} seconds, size={len(states_list)}, eval-per-second={len(states_list) / (t1 - t0):.2f}, total-batches={self.total_batches}, mean-eval-per-second={self.total_evals / self.total_time:.2f}, mean-time-per-batch={self.total_time / self.total_batches:.3f}, mean-batch-size={self.total_evals / self.total_batches:.2f}"
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
