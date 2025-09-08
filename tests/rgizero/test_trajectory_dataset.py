import numpy as np
import torch
from pathlib import Path
import pytest
from dataclasses import dataclass

from rgi.rgizero.trajectory_dataset import Vocab, TrajectoryDatasetBuilder, TrajectoryDataset, build_trajectory_loader


@dataclass
class TrajectoryDatasetFixture:
    root: Path
    split: str
    orig_actions: list[np.ndarray]
    orig_policies: list[np.ndarray]
    orig_values: list[np.ndarray]


def write_random_trajectory_dataset(
    tmp_path: Path, traj_lengths: list[int], seed: int | None = None
) -> TrajectoryDatasetFixture:
    seed = seed if seed is not None else np.random.randint(0, 1000000)
    root = tmp_path / "test_data"
    split = "train"
    vocab = Vocab(vocab_size=7, itos=list("1234567"))
    builder = TrajectoryDatasetBuilder(vocab)
    rng = np.random.default_rng(seed)

    for length in traj_lengths:
        actions = rng.integers(1, 8, size=length)  # 1-7
        policies = rng.random((length, 7), dtype=np.float32)
        values = rng.integers(-1, 2, size=length)  # -1,0,1
        builder.add_trajectory(actions, policies, values)
    builder.save(str(root), split)
    orig_actions = [builder.actions[i] for i in range(len(builder.actions))]
    orig_policies = [builder.policies[i] for i in range(len(builder.policies))]
    orig_values = [builder.values[i] for i in range(len(builder.values))]

    return TrajectoryDatasetFixture(root, split, orig_actions, orig_policies, orig_values)


@pytest.fixture
def small_dataset(tmp_path: Path) -> TrajectoryDatasetFixture:
    """Default small dataset fixture."""
    return write_random_trajectory_dataset(tmp_path, [3, 4, 4])


@pytest.fixture
def custom_dataset(tmp_path: Path, request: pytest.FixtureRequest) -> TrajectoryDatasetFixture:
    """Parametrized dataset fixture for custom configurations."""
    params = request.param
    traj_lengths = params.get("traj_lengths", [3, 4, 4])
    return write_random_trajectory_dataset(tmp_path, traj_lengths)


@pytest.mark.parametrize("block_size", [5, 10, 15, 20])
def test_save_and_load_content(small_dataset, block_size):
    ds = TrajectoryDataset(small_dataset.root, small_dataset.split, block_size)

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
    ds = TrajectoryDataset(custom_dataset.root, custom_dataset.split, block_size)

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
    ds = TrajectoryDataset(custom_dataset.root, custom_dataset.split, block_size)

    for i in range(len(ds)):
        t = ds.read_trajectory(i, apply_padding=False)
        assert t.action.shape[0] == len(custom_dataset.orig_actions[i])
        assert torch.equal(t.action, torch.from_numpy(custom_dataset.orig_actions[i]))
        assert torch.allclose(t.policy, torch.from_numpy(custom_dataset.orig_policies[i]))
        assert torch.equal(t.value, torch.from_numpy(custom_dataset.orig_values[i]))


@pytest.mark.parametrize("batch_size, workers", [(1, 0), (2, 2)])
@pytest.mark.parametrize("custom_dataset", [{"traj_lengths": [3, 3, 3, 3]}], indirect=True)
def test_dataloader_batching(custom_dataset, batch_size, workers):
    block_size = 5
    loader = build_trajectory_loader(
        custom_dataset.root,
        custom_dataset.split,
        block_size,
        batch_size=batch_size,
        device_is_cuda=False,
        workers=workers,
    )

    # Collect all batches to verify we get the expected number of items
    all_actions = []
    all_policies = []
    all_values = []

    for batch in loader:
        actions, policies, values = batch
        assert actions.shape == (batch_size, block_size)
        assert policies.shape == (batch_size, block_size, 7)
        assert values.shape == (batch_size, block_size)

        all_actions.append(actions)
        all_policies.append(policies)
        all_values.append(values)

    # Verify we got the expected number of trajectories
    total_items = sum(batch.shape[0] for batch in all_actions)
    assert total_items == len(custom_dataset.orig_actions)


@pytest.mark.parametrize("custom_dataset", [{"block_size": 2}], indirect=True)
def test_small_block_size(custom_dataset):
    block_size = 2
    ds = TrajectoryDataset(custom_dataset.root, custom_dataset.split, block_size)
    item = ds[0]
    assert item.action.shape[0] == 2  # Truncated from 3
    assert torch.equal(item.action, torch.from_numpy(custom_dataset.orig_actions[0][:2]))
