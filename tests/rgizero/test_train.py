import pytest
import torch
from rgi.rgizero.train import TrainConfig, Trainer
from rgi.rgizero.models.transformer import TransformerConfig
from rgi.rgizero.models.token_transformer import TokenTransformer
from rgi.rgizero.data.text_dataset import SimpleTextDataset
from torch.utils.data import DataLoader


class TestTrainConfig:
    def test_default_config(self):
        """Test default training configuration."""
        config = TrainConfig(model_name="test", model_version="v1")

        assert config.model_name == "test"
        assert config.model_version == "v1"
        assert config.batch_size == 12
        assert config.max_iters == 5000  # Actual default
        assert config.learning_rate == 6e-4
        assert config.device == "cuda"
        assert config.dtype == "bfloat16"

    def test_config_replace(self):
        """Test configuration replacement."""
        config = TrainConfig(model_name="test", model_version="v1", max_iters=1000)
        new_config = config.__replace__(learning_rate=1e-3)

        assert new_config.max_iters == 1000
        assert new_config.learning_rate == 1e-3
        assert config.learning_rate != new_config.learning_rate  # Original unchanged


class TestTrainer:
    @pytest.fixture
    def simple_setup(self):
        """Create a simple setup for testing."""
        # Small model config
        model_config = TransformerConfig(n_embd=32, n_layer=2, n_head=2, n_max_context=16)
        vocab_size = 20
        model = TokenTransformer(model_config, vocab_size)

        # Simple dataset
        text = "hello world test data for training"
        dataset = SimpleTextDataset(text, block_size=8, device="cpu")
        train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Training config
        train_config = TrainConfig(
            model_name="test-model",
            model_version="v1",
            max_iters=5,
            eval_interval=3,
            eval_iters=2,
            log_interval=2,
            device="cpu",
            compile=False,
            always_save_checkpoint=False,
        )

        return {"model": model, "train_config": train_config, "train_loader": train_loader, "val_loader": val_loader}

    def test_trainer_initialization(self, simple_setup):
        """Test trainer initialization."""
        trainer = Trainer(
            model=simple_setup["model"],
            train_config=simple_setup["train_config"],
            train_loader=simple_setup["train_loader"],
            val_loader=simple_setup["val_loader"],
            device="cpu",
        )

        assert trainer.model is simple_setup["model"]
        assert trainer.train_config is simple_setup["train_config"]
        assert trainer.iter_num == 0
        assert hasattr(trainer, "optimizer")

    def test_estimate_loss(self, simple_setup):
        """Test loss estimation."""
        trainer = Trainer(
            model=simple_setup["model"],
            train_config=simple_setup["train_config"],
            train_loader=simple_setup["train_loader"],
            val_loader=simple_setup["val_loader"],
            device="cpu",
        )

        with torch.no_grad():
            losses = trainer.estimate_loss()

        assert "train" in losses
        assert "val" in losses
        assert isinstance(losses["train"], float)
        assert isinstance(losses["val"], float)

    def test_training_step_basics(self, simple_setup):
        """Test basic training step functionality."""
        trainer = Trainer(
            model=simple_setup["model"],
            train_config=simple_setup["train_config"],
            train_loader=simple_setup["train_loader"],
            val_loader=simple_setup["val_loader"],
            device="cpu",
        )

        # Test that we can get a batch and compute loss
        initial_iter = trainer.iter_num
        assert initial_iter == 0

        # Test that estimate_loss works
        losses = trainer.estimate_loss()
        assert "train" in losses
        assert "val" in losses

    def test_get_lr(self, simple_setup):
        """Test learning rate scheduling."""
        trainer = Trainer(
            model=simple_setup["model"],
            train_config=simple_setup["train_config"],
            train_loader=simple_setup["train_loader"],
            val_loader=simple_setup["val_loader"],
            device="cpu",
        )

        # Test different iteration points
        trainer.iter_num = 0
        lr_start = trainer.get_lr(trainer.iter_num)

        trainer.iter_num = trainer.train_config.warmup_iters // 2
        lr_warmup = trainer.get_lr(trainer.iter_num)

        trainer.iter_num = trainer.train_config.warmup_iters
        lr_post_warmup = trainer.get_lr(trainer.iter_num)

        # During warmup, LR should increase
        assert lr_warmup > lr_start
        assert lr_post_warmup >= lr_warmup

    def test_optimizer_exists(self, simple_setup):
        """Test that trainer has an optimizer."""
        trainer = Trainer(
            model=simple_setup["model"],
            train_config=simple_setup["train_config"],
            train_loader=simple_setup["train_loader"],
            val_loader=simple_setup["val_loader"],
            device="cpu",
        )

        # Should have an optimizer
        assert hasattr(trainer, "optimizer")
        assert hasattr(trainer.optimizer, "step")
        assert hasattr(trainer.optimizer, "zero_grad")

        # Should have parameters
        assert len(trainer.optimizer.param_groups) > 0
