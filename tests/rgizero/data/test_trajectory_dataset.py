import numpy as np
import torch
from pathlib import Path
import pytest
from dataclasses import dataclass

from rgi.rgizero.common import TOKENS

from rgi.rgizero.data.trajectory_dataset import (
    Vocab,
    TrajectoryDatasetBuilder,
    TrajectoryDataset,
    build_trajectory_loader,
)


@dataclass
class TrajectoryDatasetFixture:
    root: Path
    split: str
    orig_actions: list[np.ndarray]
    orig_policies: list[np.ndarray]
    orig_values: list[np.ndarray]


def write_random_trajectory_dataset(
    tmp_path: Path, traj_lengths: list[int], seed: int | None = None, shuffle: bool = False
) -> TrajectoryDatasetFixture:
    from rgi.rgizero.games import game_registry

    seed = seed if seed is not None else np.random.randint(0, 1000000)
    root = tmp_path / "test_data"
    split = "train"

    # Create the game to get valid actions and state
    game = game_registry.create_game("connect4")
    # TODO: Can we not include start_token in vocab??
    vocab = Vocab(itos=[TOKENS.START_OF_GAME] + list(game.all_actions()))
    builder = TrajectoryDatasetBuilder(vocab)
    rng = np.random.default_rng(seed)

    for length in traj_lengths:
        state = game.initial_state()

        # Lists to store trajectory data
        actions_list = []
        policies_list = []
        values_list = []

        for _ in range(length):
            legal_actions = game.legal_actions(state)
            action = rng.choice(legal_actions)

            # TODO: Can we not include start_token in vocab??
            policy = rng.random(len(vocab.itos))
            policy = policy / policy.sum()

            # TODO: Do these sum to 0 or 1??
            value = rng.random(2)

            state = game.next_state(state, action)

            actions_list.append(action)
            policies_list.append(policy)
            values_list.append(value)

        actions_encoded = vocab.encode(actions_list)

        policies = np.stack(policies_list, axis=0).astype(np.float32)
        values = np.stack(values_list, axis=0).astype(np.float32)

        builder.add_trajectory(np.array(actions_encoded, dtype=np.int32), policies, values)

    builder.save(str(root), split, shuffle=shuffle)
    orig_actions = [builder.actions[i] for i in range(len(builder.actions))]
    orig_policies = [builder.fixed_width_policies[i] for i in range(len(builder.fixed_width_policies))]
    orig_values = [builder.values[i] for i in range(len(builder.values))]

    return TrajectoryDatasetFixture(root, split, orig_actions, orig_policies, orig_values)


@pytest.fixture
def small_dataset(tmp_path: Path) -> TrajectoryDatasetFixture:
    """Default small dataset fixture."""
    return write_random_trajectory_dataset(tmp_path, [4, 5, 5])


@pytest.fixture
def custom_dataset(tmp_path: Path, request: pytest.FixtureRequest) -> TrajectoryDatasetFixture:
    """Parametrized dataset fixture for custom configurations."""
    params = request.param
    traj_lengths = params.get("traj_lengths", [3, 4, 4])
    return write_random_trajectory_dataset(tmp_path, traj_lengths)


@pytest.mark.parametrize("block_size", [5, 10, 15, 20])
def test_save_and_load_content(small_dataset, block_size):
    ds = TrajectoryDataset(small_dataset.root, small_dataset.split, block_size, prepend_start_token=False)

    assert len(ds) == len(small_dataset.orig_actions)

    for i in range(len(ds)):
        item = ds[i]
        traj_len = len(small_dataset.orig_actions[i])
        # Content matches up to min(traj_len, block_size)
        assert torch.equal(item.action[:traj_len], torch.from_numpy(small_dataset.orig_actions[i]))
        assert torch.allclose(item.policy[:traj_len], torch.from_numpy(small_dataset.orig_policies[i]))
        assert torch.equal(item.value[:traj_len], torch.from_numpy(small_dataset.orig_values[i]))
        # Padding is zero if traj_len < block_size
        if traj_len < block_size:
            assert torch.all(item.action[traj_len:] == 0)
            assert torch.all(item.policy[traj_len:] == 0)
            assert torch.all(item.value[traj_len:] == 0)


@pytest.mark.parametrize("custom_dataset", [{"traj_lengths": [10, 5]}], indirect=True)
def test_truncation_for_long_trajectories(custom_dataset):
    block_size = 7
    ds = TrajectoryDataset(custom_dataset.root, custom_dataset.split, block_size, prepend_start_token=False)

    # First trajectory longer than block_size -> truncated
    item0 = ds[0]
    assert item0.action.shape[0] == block_size
    assert torch.equal(item0.action, torch.from_numpy(custom_dataset.orig_actions[0][:block_size]))

    # Second shorter -> padded
    item1 = ds[1]
    assert item1.action.shape[0] == block_size
    assert torch.equal(item1.action[:5], torch.from_numpy(custom_dataset.orig_actions[1]))
    assert torch.all(item1.action[5:] == 0)


@pytest.mark.parametrize("custom_dataset", [{"traj_lengths": [10, 5]}], indirect=True)
def test_get_trajectory_exact_content_unpadded(custom_dataset):
    block_size = 15
    ds = TrajectoryDataset(custom_dataset.root, custom_dataset.split, block_size, prepend_start_token=False)

    for i in range(len(ds)):
        t = ds.read_trajectory(i, apply_padding=False)
        assert t.action.shape[0] == len(custom_dataset.orig_actions[i])
        assert torch.equal(t.action, torch.from_numpy(custom_dataset.orig_actions[i]))
        assert torch.allclose(t.policy, torch.from_numpy(custom_dataset.orig_policies[i]))
        assert torch.equal(t.value, torch.from_numpy(custom_dataset.orig_values[i]))


# Note: This test can be slow when workers > 0 on some platforms (e.g. OSX).
@pytest.mark.parametrize("batch_size, workers", [(1, 0), (2, 0)])
@pytest.mark.parametrize("custom_dataset", [{"traj_lengths": [3, 3, 3, 3, 7, 3, 4, 2]}], indirect=True)
def test_dataloader_batching(custom_dataset, batch_size, workers):
    block_size = 5
    train_loader, val_loader = build_trajectory_loader(
        [custom_dataset.root / custom_dataset.split],
        block_size,
        batch_size=batch_size,
        workers=workers,
        val_split_prop=0.25,
    )

    # Collect all batches to verify we get the expected number of items
    all_actions = []
    all_policies = []
    all_values = []

    for loader in [train_loader, val_loader]:
        for actions, policies, values, padding_masks in loader:
            # TODO: This test assumes len(loader) % batch_size == 0
            assert actions.shape == (batch_size, block_size)
            assert policies.shape == (batch_size, block_size, 8)
            assert values.shape == (batch_size, block_size, 2)
            assert padding_masks.shape == (batch_size, block_size)

            all_actions.append(actions)
            all_policies.append(policies)
            all_values.append(values)

    # Verify we got the expected number of trajectories
    total_items = sum(batch.shape[0] for batch in all_actions)
    assert total_items == len(custom_dataset.orig_actions)


@pytest.mark.parametrize("custom_dataset", [{"block_size": 2}], indirect=True)
def test_small_block_size(custom_dataset):
    block_size = 2
    ds = TrajectoryDataset(custom_dataset.root, custom_dataset.split, block_size, prepend_start_token=False)
    item = ds[0]
    assert item.action.shape[0] == 2  # Truncated from 3
    assert torch.equal(item.action, torch.from_numpy(custom_dataset.orig_actions[0])[:2])


def test_prepend_start_token(small_dataset):
    """Test that start token is prepended correctly."""
    # We use a custom dataset fixture but load it with prepend_start_token=True (default/forced in modern code)
    # The dataset fixture writes data WITHOUT start tokens (length 4, 5, 5)

    # We need to manually construct the dataset to force the flag validation
    # Note: The current implementation defaults prepend_start_token logic inside read_trajectory
    # but the constructor accepts it.

    ds = TrajectoryDataset(small_dataset.root, small_dataset.split, block_size=10, prepend_start_token=True)

    # Read first item
    item = ds[0]

    # Check start token at index 0
    assert item.action[0].item() == ds.vocab.stoi[TOKENS.START_OF_GAME]

    # Check content shifted by 1
    # Original action[0] should be at index 1
    orig_act_0 = small_dataset.orig_actions[0][0]
    assert item.action[1].item() == orig_act_0

    # Verify length is preserved (original 4 -> new 4, last dropped)
    actual_len = 4
    # Check that mask accounts for actual length
    # Note: padding_mask is True for content, False for padding
    assert item.padding_mask is not None
    assert torch.sum(item.padding_mask).item() == actual_len
