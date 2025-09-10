from pathlib import Path
from contextlib import nullcontext
import torch
import os


def _find_project_root(start_path: Path | None = None, marker: str = ".git") -> Path:
    """Find project root by looking for .git dir"""
    start_path = start_path or Path.cwd()

    current = start_path.resolve()

    while current != current.parent:  # Stop at filesystem root
        if (current / marker).exists():
            return current
        current = current.parent

    # Fallback to current directory if no markers found
    return start_path


PROJECT_ROOT = _find_project_root()
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_ROOT = PROJECT_ROOT / "models"
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(MODEL_ROOT, exist_ok=True)


def model_dir(model_name: str, model_version: str) -> Path:
    dir = MODEL_ROOT / model_name / model_version
    os.makedirs(dir, exist_ok=True)
    return dir


def data_dir(name: str) -> Path:
    dir = DATA_ROOT / name
    os.makedirs(dir, exist_ok=True)
    return dir


def get_ctx(dtype: str, device: str):
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    return nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
