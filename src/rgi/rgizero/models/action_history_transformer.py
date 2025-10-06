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
        action_history = state.action_history
        history_tokens = self.vocab.encode(action_history)

        # action_history tokens already in [0..7]; we feed last block_size tokens
        x = torch.tensor(history_tokens[-self.block_size :], dtype=torch.long, device=self.device)[None, :]
        # simple pad-left with zeros (new_game tokens) if needed
        if x.size(1) < self.block_size:
            pad = torch.zeros((1, self.block_size - x.size(1)), dtype=torch.long, device=self.device)
            x = torch.cat([pad, x], dim=1)
        (policy_logits, value_logits), _ = self.model(x)
        # policy for next move: softmax over vocab (8)
        policy = F.softmax(policy_logits[0], dim=-1)
        # value: expectation over {-1,0,1}
        val_probs = F.softmax(value_logits[0], dim=-1)
        value = val_probs * 2 - 1  # Convert from probabilities to [-1,1] domain.

        policy_np = policy.cpu().numpy()
        value_np = value.cpu().numpy()
        legal_policy_idx = np.array([self.vocab.stoi[a] for a in legal_actions])
        legal_policy = policy_np[legal_policy_idx]

        return NetworkEvaluatorResult(legal_policy, value_np)
