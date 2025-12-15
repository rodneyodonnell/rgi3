import pytest
import torch
import numpy as np

from rgi.rgizero.models.action_history_transformer import ActionHistoryTransformer, ActionHistoryTransformerEvaluator
from rgi.rgizero.models.transformer import TransformerConfig
from rgi.rgizero.games.connect4 import Connect4Game
from rgi.rgizero.games.history_wrapper import HistoryTrackingGame
from rgi.rgizero.data.trajectory_dataset import Vocab
from rgi.rgizero.common import TOKENS


@pytest.fixture
def transformer_config():
    return TransformerConfig(n_embd=16, n_layer=2, n_head=4, n_max_context=8)


@pytest.fixture
def vocab(game):
    return Vocab(itos=[TOKENS.START_OF_GAME] + list(game.base_game.all_actions()))


@pytest.fixture
def action_history_transformer(transformer_config, vocab):
    action_vocab_size = vocab.vocab_size
    num_players = 2
    return ActionHistoryTransformer(transformer_config, action_vocab_size, num_players)


@pytest.fixture
def game():
    return HistoryTrackingGame(Connect4Game(connect_length=5))


@pytest.fixture
def action_history_transformer_evaluator(action_history_transformer, vocab):
    device = "cpu"
    n_max_context = action_history_transformer.config.n_max_context
    return ActionHistoryTransformerEvaluator(
        action_history_transformer, device=device, block_size=n_max_context, vocab=vocab
    )


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

        # normalize
        policy_target_row = (policy_target_row / policy_target_row.sum(axis=1).unsqueeze(1))
        value_target_row = (value_target_row / value_target_row.sum(axis=1).unsqueeze(1))

        def get_loss(padding_mask):
            batch_size = padding_mask.shape[0] if padding_mask is not None else 1

            (policy_logits, value_logits), loss_dict, loss = model(
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

        (policy_logits, value_logits), loss_dict, loss = model(idx)

        assert policy_logits.shape == (batch_size, 1, vocab_size)
        assert value_logits.shape == (batch_size, 1, num_players)
        assert loss is None

    def test_forward_inference_mode_with_len(self, action_history_transformer):
        model = action_history_transformer
        batch_size, seq_len = 3, 6
        vocab_size = model.action_vocab_size
        num_players = model.num_players

        idx = torch.randint(0, vocab_size, (batch_size, seq_len))

        encoded_len_1 = torch.tensor([2,4,2])
        (policy_logits_1, value_logits_1), loss_dict_1, loss_1 = model(idx, encoded_len=encoded_len_1)

        assert policy_logits_1.shape == (batch_size, 1, vocab_size)
        assert value_logits_1.shape == (batch_size, 1, num_players)
        assert loss_1 is None

        encoded_len_2 = torch.tensor([3,4,6])
        (policy_logits_2, value_logits_2), _, _ = model(idx, encoded_len=encoded_len_2)

        assert torch.all(policy_logits_1[0] != policy_logits_2[0])
        assert torch.all(policy_logits_1[1] == policy_logits_2[1])
        assert torch.all(policy_logits_1[2] != policy_logits_2[2])

        assert torch.all(value_logits_1[0] != value_logits_2[0])
        assert torch.all(value_logits_1[1] == value_logits_2[1])
        assert torch.all(value_logits_1[2] != value_logits_2[2])

        # _last is the last element, so expect logits_2[2] == logits_last[2] 
        (policy_logits_last, value_logits_last), _, _ = model(idx)
        assert torch.all(policy_logits_last[2] != policy_logits_1[2])
        assert torch.all(value_logits_last[2] != value_logits_1[2])
        assert torch.all(policy_logits_last[2] == policy_logits_2[2])
        assert torch.all(value_logits_last[2] == value_logits_2[2])


class TestActionHistoryTransformerEvaluator:
    def test_evaluate(self, action_history_transformer_evaluator, game):
        evaluator = action_history_transformer_evaluator
        state = game.initial_state()
        legal_actions = game.legal_actions(state)

        result = evaluator.evaluate(game, state, legal_actions)

        assert result.legal_policy.shape == (len(legal_actions),)
        assert result.player_values.shape == (game.num_players(state),)
        assert np.isclose(np.sum(result.legal_policy), 1.0)
        assert result.legal_policy.dtype == np.float32
        assert result.player_values.dtype == np.float32

    def test_evaluate_batch(self, action_history_transformer_evaluator, game):
        evaluator = action_history_transformer_evaluator

        num_states = 4
        states = []
        legal_actions_list = []
        state = game.initial_state()
        for _ in range(num_states):
            states.append(state)
            legal_actions_list.append(game.legal_actions(state))
            action = game.all_actions()[0]  # Take a dummy action to advance state
            state = game.next_state(state, action)

        results = evaluator.evaluate_batch(game, states, legal_actions_list)

        assert len(results) == num_states
        for i, result in enumerate(results):
            assert result.legal_policy.shape == (len(legal_actions_list[i]),)
            assert result.player_values.shape == (game.num_players(states[i]),)
            assert np.isclose(np.sum(result.legal_policy), 1.0)
            assert result.legal_policy.dtype == np.float32
            assert result.player_values.dtype == np.float32
