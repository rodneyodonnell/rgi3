"""
Multitask transformer from action-histyry to Policy and Value heads.
"""

from typing import override, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rgi.rgizero.models.transformer import TransformerConfig, Transformer, LayerNorm, init_weights
from rgi.rgizero.models.token_transformer import TokenTransformer
from rgi.rgizero.data.trajectory_dataset import Vocab
from rgi.rgizero.players.alphazero import NetworkEvaluator, NetworkEvaluatorResult


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
    ):
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
                # Calcualte the average loss per unpadded tokens.
                # note: We may want to experiment with average per batch?
                policy_loss = F.cross_entropy(flat_policy_logits, flat_policy_target, reduction="mean")
                loss += policy_loss

            if value_target is not None:
                flat_value_logits = value_logits.view(-1, self.num_players)
                flat_value_target = value_target.view(-1, self.num_players)
                # Mask out padding tokens so they are not considered in the loss or gradients.
                if flat_padding_mask is not None:
                    flat_value_logits = flat_value_logits[flat_padding_mask]
                    flat_value_target = flat_value_target[flat_padding_mask]
                value_loss = F.cross_entropy(flat_value_logits, flat_value_target, reduction="mean")
                loss += value_loss
        else:
            # Inference mode: only compute logits for final position
            h_last = h[:, -1, :]  # (B, n_embd)
            policy_logits, value_logits = self.policy_value_head(h_last)
            loss = None

        return (policy_logits, value_logits), loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return TokenTransformer.configure_optimizers(self, weight_decay, learning_rate, betas, device_type)


class ActionHistoryTransformerEvaluator(NetworkEvaluator):
    """Neural network evaluator for MCTS."""

    def __init__(self, model: ActionHistoryTransformer, device: str, block_size: int, vocab: Vocab):
        self.model = model.eval()
        self.device = device
        self.block_size = block_size
        self.vocab = vocab

    @override
    @torch.no_grad()
    def evaluate(self, game, state, legal_actions) -> NetworkEvaluatorResult:
        return self.evaluate_batch([state], [legal_actions])[0]

    @torch.inference_mode()
    def evaluate_batch(self, states_list, legal_actions_list):
        B = len(states_list)
        L = self.block_size

        # Start with zeros to simplify padding.
        x_np = np.zeros((B, L), dtype=np.int32)
        for i, state in enumerate(states_list):
            encoded_row = self.vocab.encode(state.action_history)[-self.block_size :]
            x_np[i, L - len(encoded_row) :] = encoded_row
        x_pinned = torch.from_numpy(x_np).pin_memory()

        # Process model on GPU.
        x_gpu = x_pinned.to(self.device, non_blocking=True)
        (policy_logits_gpu, value_logits_gpu), _ = self.model(x_gpu)

        # Calculate legal_policy_mask on CPU
        legal_policy_mask_np = np.zeros(policy_logits_gpu.shape, dtype=np.bool_)
        for i, legal_actions in enumerate(legal_actions_list):
            legal_idx = np.array([self.vocab.stoi[a] for a in legal_actions], dtype=np.int32)
            legal_policy_mask_np[i, legal_idx] = True
        legal_policy_mask_pinned = torch.from_numpy(legal_policy_mask_np).pin_memory()

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
        return ret
