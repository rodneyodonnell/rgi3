from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rgi.rgizero.models.transformer import TransformerConfig, Transformer, LayerNorm, init_weights
from rgi.rgizero.models.token_transformer import TokenTransformer
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
