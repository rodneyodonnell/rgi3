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
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass
from typing import Iterable, Any


@dataclass
class TrajectoryTuple:
    action: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor


@dataclass
class Vocab:
    """Vocabulary helper with encode/decode and file (de)serialization."""

    vocab_size: int
    itos: list[str]
    stoi: dict[str, int]

    def __init__(self, vocab_size: int, itos: list[str] | None = None, stoi: dict[str, int] | None = None):
        assert vocab_size > 0, "Vocab size must be greater than 0"
        if itos is None:
            if stoi is None:
                stoi = {str(i): i for i in range(vocab_size)}
            itos = [str(i) for i in range(vocab_size)]
        assert len(itos) == vocab_size, f"itos must have {vocab_size} items"

        if stoi is None:
            stoi = {token: idx for idx, token in enumerate(itos)}
        assert len(stoi) == vocab_size, f"stoi must have {vocab_size} items"

        self.vocab_size = vocab_size
        self.itos = list(itos)
        self.stoi = dict(stoi)

    @classmethod
    def from_dict(cls, meta: dict) -> "Vocab":
        return cls(meta["vocab_size"], meta.get("itos"), meta.get("stoi"))

    def to_dict(self) -> dict:
        return {"vocab_size": self.vocab_size, "itos": self.itos, "stoi": self.stoi}

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.stoi[token] for token in tokens]

    def decode(self, indices: Iterable[int]) -> str:
        return "".join(self.itos[idx] for idx in indices)


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
        self.policies: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        self.vocab = vocab

    def add_trajectory(self, actions: np.ndarray, policies: np.ndarray, values: np.ndarray):
        """Add a trajectory to the dataset."""
        assert actions.shape == values.shape
        assert actions.shape[0] == policies.shape[0]
        assert policies.shape[1] == self.vocab.vocab_size
        self.actions.append(actions)
        self.policies.append(policies)
        self.values.append(values)

    def save(self, root_dir: str, split: str):
        split_dir = pathlib.Path(root_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(split_dir / "vocab.json", "w") as f:
            vocab_dict = self.vocab.to_dict()
            json.dump(vocab_dict, f)

        action_lengths = [len(action) for action in self.actions]
        boundaries = np.cumsum([0] + action_lengths).astype(np.int64)
        if boundaries[-1] < 2**31:
            boundaries = boundaries.astype(np.int32)

        # Save data
        np.save(split_dir / "action.npy", np.concatenate(self.actions))
        np.save(split_dir / "policy.npy", np.concatenate(self.policies))
        np.save(split_dir / "value.npy", np.concatenate(self.values))
        np.save(split_dir / "boundaries.npy", boundaries.astype(np.int64))


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
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open_memmaps()

    def __len__(self) -> int:
        return self._num_trajectories

    def __getitem__(self, trajectory_idx: int) -> TrajectoryTuple:
        return self.read_trajectory(trajectory_idx, apply_padding=True)

    def read_trajectory(self, trajectory_idx: int, apply_padding: bool) -> TrajectoryTuple:
        action_start_idx = self.boundaries[trajectory_idx]
        action_end_idx = self.boundaries[trajectory_idx + 1]

        action = torch.from_numpy(self.action_data[action_start_idx:action_end_idx])
        policy = torch.from_numpy(self.policy_data[action_start_idx:action_end_idx])
        value = torch.from_numpy(self.value_data[action_start_idx:action_end_idx])

        if apply_padding:
            pad_len = self.block_size - (action_end_idx - action_start_idx)
            action = torch.nn.functional.pad(action, (0, pad_len))
            policy = torch.nn.functional.pad(policy, (0, 0, 0, pad_len))
            value = torch.nn.functional.pad(value, (0, pad_len))
        return TrajectoryTuple(action, policy, value)

    def _read_vocab(self) -> Vocab:
        with open(self.split_dir / "vocab.json", "r") as f:
            vocab_dict = json.load(f)
        return Vocab.from_dict(vocab_dict)


def trajectory_collate_fn(batch: list[TrajectoryTuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom collate function for TrajectoryTuple objects."""
    actions = torch.stack([item.action for item in batch])
    policies = torch.stack([item.policy for item in batch])
    values = torch.stack([item.value for item in batch])
    return actions, policies, values


def build_trajectory_loader(
    root_dir: str | pathlib.Path,
    split: str,
    block_size: int,
    batch_size: int,
    device_is_cuda: bool,
    shuffle: bool = True,
    workers: int = 4,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    ds = TrajectoryDataset(root_dir, split, block_size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device_is_cuda,
        collate_fn=trajectory_collate_fn,
    )  # type: ignore
