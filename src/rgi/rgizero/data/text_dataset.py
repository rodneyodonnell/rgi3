import torch
from torch.utils.data import Dataset


class SimpleTextDataset(Dataset):
    def __init__(self, text: str, block_size: int, chars: str | list[str] | None = None, device: str = "cpu"):
        self.block_size = block_size
        self.device = device

        if chars is None:
            chars = sorted(list(set(text)))

        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: list[str] = chars
        self.vocab_size = len(chars)

        # Note: We place the entire dataset on the specified device.
        #       This may cause memory issues for large datasets.
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long, device=device)

        print(f"Dataset size: {len(self.data)} characters")
        print(f"Vocab size: {self.vocab_size}")
        print(f"Unique chars: {''.join(chars)}")
        print(f"Data stored on device: {self.data.device}")

    def __len__(self) -> int:
        """Return number of possible sequences."""
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sequence."""
        # Note: self.data is already on self.device, so we don't need to move it here.
        x = self.data[idx : idx + self.block_size]  # input sequence
        y = self.data[idx + 1 : idx + self.block_size + 1]  # target sequence (shifted by 1)
        return x, y

    def encode(self, text: str) -> list[int]:
        """Encode text to list of token indices."""
        return [self.stoi[c] for c in text]

    def decode(self, tokens: list[int]) -> str:
        """Decode list of token indices to text."""
        return "".join([self.itos[i] for i in tokens])
