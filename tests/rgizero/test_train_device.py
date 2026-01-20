import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from rgi.rgizero.train import Trainer, TrainConfig
import pytest


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        out = self.linear(x)
        loss = out.mean()
        return out, {"loss": loss}, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return torch.optim.SGD(self.parameters(), lr=learning_rate)


def get_available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.mark.parametrize("device", get_available_devices())
def test_trainer_handles_device_mismatch(device):
    """
    Test that the Trainer correctly moves input data to the model's device
    (works for CPU and CUDA).
    """
    # Setup
    model = SimpleModel().to(device)

    # Create dummy data on CPU
    x = torch.randn(10, 10)  # on CPU by default
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=2)

    config = TrainConfig(
        model_name="test_model",
        model_version="v0",
        device=device,
        max_epochs=1,
        max_iters=2,
        eval_interval=1,
        eval_iters=1,
        gradient_accumulation_steps=1,
        wandb_log=False,
        always_save_checkpoint=False,  # simplify
    )

    trainer = Trainer(
        model=model,
        train_config=config,
        train_loader=loader,
        val_loader=loader,
        device=device,
        model_dir="/tmp/test_model_dir",  # Dummy dir
    )

    # Run estimate_loss - this would fail if data isn't moved
    try:
        trainer.estimate_loss()
    except RuntimeError as e:
        pytest.fail(f"estimate_loss failed with error: {e}")

    # Run one training step via train_epoch implies running part of train loop
    # We can just call train() for a very short duration
    try:
        trainer.train()
    except RuntimeError as e:
        pytest.fail(f"train() failed with error: {e}")
