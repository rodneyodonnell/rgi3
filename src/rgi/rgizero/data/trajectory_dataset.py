"""
Datastet loading & saving for 'Trajectory' and 'Token'  based training.

A Trajectory is a sequence of actions, policies, and values.
- Actions are the actions taken by the agent.
- Policies are the policies (distribution over actions)used by the agent.
- Values are the expected (or actual) final values of a trajectory.
"""

import pathlib
import numpy as np
import json
import torch
import warnings
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass
from typing import Any, Sequence

from rgi.rgizero.common import TOKENS

@dataclass
class TrajectoryTuple:
    action: torch.Tensor  # (L, num_actions)
    policy: torch.Tensor  # (L, num_actions)
    value: torch.Tensor  # (L, num_players)
    padding_mask: torch.Tensor | None  # (L,)


# Token can be any hashable type? Restrict to `str | int` for now.
Token = str | int


@dataclass
class Vocab:
    """Vocabulary helper with encode/decode and file (de)serialization."""

    vocab_size: int
    itos: Sequence[Token]
    stoi: dict[Token, int]

    def __init__(self, itos: Sequence[Token]):
        stoi = {token: idx for idx, token in enumerate(itos)}

        self.itos = itos
        self.stoi = stoi
        self.vocab_size = len(itos)

    @classmethod
    def from_dict(cls, meta: dict) -> "Vocab":
        itos: Sequence[Token] = meta.get("itos")  # type: ignore
        return Vocab(itos)

    def to_dict(self) -> dict:
        return {"vocab_size": self.vocab_size, "itos": self.itos, "stoi": self.stoi}

    def encode(self, tokens: Sequence[Token]) -> Sequence[int]:
        return np.array([self.stoi[token] for token in tokens])

    def decode(self, indices: Sequence[int]) -> Sequence[Token]:
        return [self.itos[idx] for idx in indices]

    def decode_str(self, indices: Sequence[int]) -> str:
        return "".join(str(token) for token in self.decode(indices))


class TrajectoryDatasetBuilder:
    """Dataset builder for Trajectory-style trajectories.

    Usage:
        vocab = Vocab(vocab_size=7, itos=list("1234567"))
        builder = TrajectoryDatasetBuilder(vocab)
        builder.add_trajectory(actions, policies, values)
        builder.save(root_dir, split)
    """

    def __init__(self, vocab: Vocab):
        self.actions: list[np.ndarray] = []
        self.fixed_width_policies: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        self.vocab = vocab

    def add_trajectory(self, actions: np.ndarray, fixed_width_policies: np.ndarray, values: np.ndarray):
        """Add a trajectory to the dataset."""
        assert actions.shape[0] == values.shape[0], (
            f"actions.shape[0] != values.shape[0]: {actions.shape}[0] != {values.shape}[0]"
        )
        assert actions.shape[0] == fixed_width_policies.shape[0], (
            f"actions.shape[0] != policies.shape[0]: {actions.shape}[0] != {fixed_width_policies.shape}[0]"
        )
        assert fixed_width_policies.shape[1] == (self.vocab.vocab_size), (
            f"policies.shape[1] != vocab.vocab_size: {fixed_width_policies.shape}[1] != {self.vocab.vocab_size}"
        )
        self.actions.append(actions)
        self.fixed_width_policies.append(fixed_width_policies)
        self.values.append(values)

    def save(self, root_dir: str, split: str, shuffle: bool = True) -> str:
        """Save trajectories to disk, returns path to split directory."""
        split_dir = pathlib.Path(root_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(split_dir / "vocab.json", "w") as f:
            vocab_dict = self.vocab.to_dict()
            json.dump(vocab_dict, f)

        if shuffle:
            # shuffle in builder so we don't need to do it in DataLoader.
            indices = np.random.permutation(len(self.actions))
            self.actions = [self.actions[i] for i in indices]
            self.fixed_width_policies = [self.fixed_width_policies[i] for i in indices]
            self.values = [self.values[i] for i in indices]

        action_lengths = [len(action) for action in self.actions]
        boundaries = np.cumsum([0] + action_lengths).astype(np.int64)
        if boundaries[-1] < 2**31:
            boundaries = boundaries.astype(np.int32)

        # Save data
        np.save(split_dir / "action.npy", np.concatenate(self.actions))
        np.save(split_dir / "policy.npy", np.concatenate(self.fixed_width_policies))
        np.save(split_dir / "value.npy", np.concatenate(self.values))
        np.save(split_dir / "boundaries.npy", boundaries.astype(np.int64))

        return split_dir


# TODO: Should this be a `torch.StackDataset` instead?
class TrajectoryDataset(Dataset[TrajectoryTuple]):
    """Dataset for reading trajectory datasets."""

    action_data: np.ndarray
    policy_data: np.ndarray
    value_data: np.ndarray
    boundaries: np.ndarray

    def __init__(self, root_dir: str | pathlib.Path, split: str, block_size: int) -> None:
        self.split_dir = pathlib.Path(root_dir) / split
        self.block_size = block_size

        self.vocab = self._read_vocab()
        self.start_prefix_idx = torch.tensor([self.vocab.stoi[TOKENS.START_OF_GAME]])

        # file paths for safe reopening in worker processes
        self._action_path = self.split_dir / "action.npy"
        self._policy_path = self.split_dir / "policy.npy"
        self._value_path = self.split_dir / "value.npy"
        self._boundaries_path = self.split_dir / "boundaries.npy"

        # populated in self._open_memmaps()
        self._open_memmaps()

        assert self.action_data is not None
        assert self.boundaries is not None
        self._num_trajectories: int = len(self.boundaries) - 1
        self._num_actions: int = self.action_data.shape[0]

    def _open_memmaps(self) -> None:
        self.action_data = np.load(self._action_path, mmap_mode="r")
        self.policy_data = np.load(self._policy_path, mmap_mode="r")
        self.value_data = np.load(self._value_path, mmap_mode="r")
        self.boundaries = np.load(self._boundaries_path, mmap_mode="r")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Drop memmaps before pickling for DataLoader workers
        state["action_data"] = None
        state["policy_data"] = None
        state["value_data"] = None
        state["boundaries"] = None
        state['split_dir'] = str(state['split_dir']) # Replace PosixPath with string for pickling. 
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open_memmaps()

    def __len__(self) -> int:
        return self._num_trajectories

    def __getitem__(self, trajectory_idx: int) -> TrajectoryTuple:
        return self.read_trajectory(trajectory_idx, apply_padding=True, prepend_start_token=True)

    def read_trajectory(self, trajectory_idx: int, apply_padding: bool, prepend_start_token: bool) -> TrajectoryTuple:
        action_start_idx = self.boundaries[trajectory_idx]
        action_end_idx = self.boundaries[trajectory_idx + 1]

        # Suppress warning about non-writable arrays since we only read from tensors
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable.*")
            if prepend_start_token:
                action = torch.cat([self.start_prefix_idx, torch.from_numpy(self.action_data[action_start_idx:action_end_idx-1])])
            else:
                action = torch.from_numpy(self.action_data[action_start_idx:action_end_idx])
            policy = torch.from_numpy(self.policy_data[action_start_idx:action_end_idx])
            # TODO: value_data repeatedly stores the same value... we should just store it once per trajectory.
            # For now, store per-step values to match tests and pad/truncate like actions/policies.
            value = torch.from_numpy(self.value_data[action_start_idx:action_end_idx])

        if apply_padding:
            action_len = action_end_idx - action_start_idx
            pad_len = self.block_size - action_len
            action = torch.nn.functional.pad(action, (0, pad_len))
            policy = torch.nn.functional.pad(policy, (0, 0, 0, pad_len))
            value = torch.nn.functional.pad(value, (0, 0, 0, pad_len))
            padding_mask = torch.zeros(self.block_size, dtype=torch.bool)
            padding_mask[:action_len] = True
        else:
            padding_mask = None

        return TrajectoryTuple(action, policy, value, padding_mask)

    def _read_vocab(self) -> Vocab:
        with open(self.split_dir / "vocab.json", "r") as f:
            vocab_dict = json.load(f)
        return Vocab.from_dict(vocab_dict)


def trajectory_collate_fn(
    batch: list[TrajectoryTuple],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom collate function for TrajectoryTuple objects."""
    actions = torch.stack([item.action for item in batch])
    policies = torch.stack([item.policy for item in batch])
    values = torch.stack([item.value for item in batch])
    padding_masks = torch.stack([item.padding_mask for item in batch])
    return actions, policies, values, padding_masks


def build_trajectory_loader(
    dataset_paths: list[pathlib.Path],
    block_size: int,
    batch_size: int,
    device: str | torch.device | None = None,
    workers: int = 0,
    shuffle: bool = True,
    val_split_prop: float = 0.1,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    tds = [TrajectoryDataset(path.parent, path.name, block_size=block_size) for path in dataset_paths]
    full_dataset = torch.utils.data.ConcatDataset(tds)

    val_size = int(len(full_dataset) * val_split_prop)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    if train_size == 0 or val_size == 0:
        raise ValueError(
            f"Not enough data to split into train and validation sets, train_size={train_size}, val_size={val_size}"
        )
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # Create device-aware collate function if device is specified
    if device == "mps":
        device = torch.device(device)

        def collate_fn(batch):
            actions, policies, values, padding_masks = trajectory_collate_fn(batch)
            # Copy to MPS, and convert dtype as MPS doesn't support float64.
            return (
                actions.to(device),
                policies.to(device, dtype=torch.float32),
                values.to(device, dtype=torch.float32),
                padding_masks.to(device, dtype=torch.bool),
            )

    else:
        collate_fn = trajectory_collate_fn

    # Only use pin_memory for CUDA (MPS doesn't support it)
    use_pin_memory = device is not None and not isinstance(device, str) and device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=shuffle,
        pin_memory=use_pin_memory,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(42),
        persistent_workers=(workers > 0),
    )  # type: ignore

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=shuffle,
        pin_memory=use_pin_memory,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(42),
        persistent_workers=(workers > 0),
    )  # type: ignore

    return train_loader, val_loader


from collections import Counter, defaultdict


def print_dataset_stats(
    dataset_paths: list[pathlib.Path], block_size: int, action_vocab: Vocab, model: torch.nn.Module = None, game=None,
):
    """Print statistics about a loaded trajectory dataset."""
    tds = [TrajectoryDataset(path.parent, path.name, block_size=block_size) for path in dataset_paths]
    td = torch.utils.data.ConcatDataset(tds)
    total_actions = 0

    # Model Verification setup
    evaluator = None
    if model is not None:
        model.eval()
        device = next(model.parameters()).device
        from rgi.rgizero.evaluators import ActionHistoryTransformerEvaluator        
        evaluator = ActionHistoryTransformerEvaluator(model, device, block_size, action_vocab, verbose=False)

    from collections import defaultdict
    dd_n = defaultdict(int)
    dd_win0 = defaultdict(int)
    dd_win1 = defaultdict(int)
    dd_draw = defaultdict(int)

    # Iterate over dataset
    for traj in td:
        actions = traj.action[traj.padding_mask]
        # policies = traj.policy[traj.padding_mask]
        values = traj.value[traj.padding_mask]
        traj_len = actions.size(0)

        total_actions += len(actions)
        final_values = values[-1] # tensor shape (2,)

        for key_len in range(min(3, traj_len)):
            key = tuple(a.item() for a in actions[:key_len])
            dd_n[key] += 1
            if final_values[0] > final_values[1]:
                dd_win0[key] += 1
            elif final_values[1] > final_values[0]:
                dd_win1[key] += 1
            else:
                dd_draw[key] += 1

    # Print basic stats
    print(f"Dataset Stats:")
    print(f"  Trajectories: {len(td)}")
    print(f"  Total actions: {total_actions}")
    print(f"  Avg trajectory length: {total_actions/len(td):.2f}")


    print(f"Prefix Stats:")
    min_print_key = dd_n[()] * 0.05
    all_actions = game.all_actions()
    state0 = game.initial_state()
    for key in sorted(dd_n.keys()):
        if len(key) >= 2 and dd_n[key] < min_print_key:
            # Don't print borking keys.
            continue
        win1_pct = 100 * dd_win0[key] / dd_n[key]
        if evaluator is not None:
            state = state0
            for a in key:
                state = game.next_state(state, a)
            eval_output = evaluator.evaluate(game, state, all_actions)
            model_win1_pct = 100 * ((eval_output.player_values + 1)/2)[0].item()
            model_legal_policy = eval_output.legal_policy
        else:
            model_win1_pct = None
            model_legal_policy = None
        # print(f"actions={key}: {dd_n[key]} win={dd_win0[key]} loss={dd_win1[key]} draw={dd_draw[key]} win1%={win1_pct:.2f} model-win1%={model_win1_prob:.2f} model-legal-policy={model_legal_policy}")
        print(f"actions={key}: {dd_n[key]} win={dd_win0[key]} loss={dd_win1[key]} draw={dd_draw[key]} win1%={win1_pct:.2f} model-win1%={model_win1_pct:.2f}")


