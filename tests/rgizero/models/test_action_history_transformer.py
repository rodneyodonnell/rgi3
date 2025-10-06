import pytest
import torch

from rgi.rgizero.models.action_history_transformer import (
    ActionHistoryTransformer,
)
from rgi.rgizero.models.transformer import TransformerConfig


@pytest.fixture
def transformer_config():
    return TransformerConfig(n_embd=16, n_layer=2, n_head=4, n_max_context=8)


@pytest.fixture
def action_history_transformer(transformer_config):
    action_vocab_size = 5
    num_players = 2
    return ActionHistoryTransformer(transformer_config, action_vocab_size, num_players)


class TestActionHistoryTransformer:
    def test_weight_tying(self, action_history_transformer):
        model = action_history_transformer
        assert model.policy_value_head.policy.weight is model.action_embedding.weight

    def test_loss_withpadding_mask(self, action_history_transformer):
        model = action_history_transformer
        seq_len = 3
        vocab_size = model.action_vocab_size
        num_players = model.num_players

        idx_row = torch.randint(0, vocab_size, (1, seq_len))
        policy_target_row = torch.rand(seq_len, vocab_size)
        value_target_row = torch.rand(seq_len, num_players)

        def get_loss(padding_mask):
            batch_size = padding_mask.shape[0] if padding_mask is not None else 1

            (policy_logits, value_logits), loss = model(
                torch.tile(idx_row, (batch_size, 1)),
                policy_target=torch.tile(policy_target_row, (batch_size, 1)),
                value_target=torch.tile(value_target_row, (batch_size, 1)),
                padding_mask=padding_mask,
            )
            # detach gradient to avoid sputious warnings from pytorch.
            return loss.detach()

        # Check the sum of losses is the same if we:
        # - Don't specify a padding mask (default is all unpadded)
        # - Specify all unpadded
        # - Average the loss over three calls, each with one unpadded token.
        #   The loss is the per-token loss, so we need to average here instead of summing.
        # - Include all unpadded tokens individually in a single batch.
        unpadded_loss = get_loss(padding_mask=None)
        explicit_unpadded_loss = get_loss(padding_mask=torch.tensor([[True, True, True]], dtype=torch.bool))
        split_padded_loss = [
            get_loss(padding_mask=torch.tensor([[True, False, False]], dtype=torch.bool)),
            get_loss(padding_mask=torch.tensor([[False, True, False]], dtype=torch.bool)),
            get_loss(padding_mask=torch.tensor([[False, False, True]], dtype=torch.bool)),
        ]
        batch_padded_loss = get_loss(
            padding_mask=torch.tensor(
                [[True, False, False], [False, True, False], [False, False, True]], dtype=torch.bool
            )
        )

        assert unpadded_loss > 0
        assert torch.allclose(unpadded_loss, explicit_unpadded_loss)
        assert torch.allclose(unpadded_loss, torch.mean(torch.tensor(split_padded_loss)))
        assert torch.allclose(unpadded_loss, batch_padded_loss)

    def test_forward_inference_mode(self, action_history_transformer):
        model = action_history_transformer
        batch_size, seq_len = 3, 6
        vocab_size = model.action_vocab_size
        num_players = model.num_players

        idx = torch.randint(0, vocab_size, (batch_size, seq_len))

        (policy_logits, value_logits), loss = model(idx)

        assert policy_logits.shape == (batch_size, vocab_size)
        assert value_logits.shape == (batch_size, num_players)
        assert loss is None
