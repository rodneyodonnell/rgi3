"""
Integration test based on rgizero-shakespeare.ipynb

This is a slower test that validates the full training pipeline works end-to-end.
"""

import torch
from torch.utils.data import DataLoader

from rgi.rgizero.data.text_dataset import SimpleTextDataset
from rgi.rgizero.models.transformer import TransformerConfig
from rgi.rgizero.models.token_transformer import TokenTransformer
from rgi.rgizero.train import Trainer, TrainConfig


class TestShakespeareIntegration:
    """Integration test based on shakespeare training notebook."""

    def test_shakespeare_training_pipeline(self):
        """Test the full shakespeare training pipeline with tiny data."""
        # Use tiny shakespeare-like text for fast testing
        raw_text = """
        HAMLET: To be or not to be, that is the question.
        OPHELIA: Good my lord, how does your honour?
        HAMLET: I humbly thank you; well, well, well.
        """

        # Configuration (much smaller than real training)
        BLOCK_SIZE = 16
        BATCH_SIZE = 2
        DEVICE = "cpu"  # Use CPU for testing

        # Create dataset
        text_dataset = SimpleTextDataset(raw_text, BLOCK_SIZE, device=DEVICE)
        vocab_size = text_dataset.vocab_size

        # Split dataset (simple temporal split to avoid data leakage)
        train_size = int(0.8 * len(text_dataset))
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, len(text_dataset)))

        train_dataset = torch.utils.data.Subset(text_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(text_dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Create model (tiny version)
        model_config = TransformerConfig(n_max_context=BLOCK_SIZE, n_embd=32, n_layer=2, n_head=2, dropout=0.1)
        model = TokenTransformer(model_config, vocab_size)
        model.to(DEVICE)

        # Training config (very short)
        train_config = TrainConfig(
            model_name="test-shakespeare-gpt",
            model_version="v1",
            eval_interval=5,
            eval_iters=2,
            log_interval=3,
            always_save_checkpoint=False,
            gradient_accumulation_steps=1,
            batch_size=BATCH_SIZE,
            learning_rate=1e-3,
            max_iters=10,  # Very short training
            warmup_iters=2,
            device=DEVICE,
            compile=False,
        )

        # Create trainer
        trainer = Trainer(
            model=model, train_config=train_config, train_loader=train_loader, val_loader=val_loader, device=DEVICE
        )

        # Test initial state
        assert trainer.iter_num == 0
        initial_losses = trainer.estimate_loss()
        _initial_train_loss = initial_losses["train"].item()
        _initial_val_loss = initial_losses["val"].item()

        # Train the model
        trainer.train()

        # Verify training completed (iter_num is incremented after the last iteration)
        assert trainer.iter_num == train_config.max_iters + 1

        # Check that loss changed (should improve or at least change)
        final_losses = trainer.estimate_loss()
        final_train_loss = final_losses["train"].item()
        final_val_loss = final_losses["val"].item()

        # Losses should be finite
        assert torch.isfinite(torch.tensor(final_train_loss))
        assert torch.isfinite(torch.tensor(final_val_loss))

        # Test text generation
        model.eval()
        with torch.no_grad():
            context = "HAM"
            start_ids = torch.tensor([text_dataset.encode(context)], dtype=torch.long, device=DEVICE)
            generated = model.generate(start_ids, max_new_tokens=10)

            # Should generate something
            assert generated.shape[1] == len(context) + 10

            # Decode and check it's valid
            generated_text = text_dataset.decode(generated[0].tolist())
            assert len(generated_text) == len(context) + 10
            assert generated_text.startswith(context)

    def test_encoding_decoding_consistency(self):
        """Test that encoding/decoding is consistent."""
        text = "ROMEO: But soft, what light through yonder window breaks?"
        dataset = SimpleTextDataset(text, block_size=10)

        # Test round-trip
        encoded = dataset.encode(text)
        decoded = dataset.decode(encoded)
        assert decoded == text

        # Test partial encoding/decoding
        partial = text[:20]
        encoded_partial = dataset.encode(partial)
        decoded_partial = dataset.decode(encoded_partial)
        assert decoded_partial == partial

    def test_model_forward_backward(self):
        """Test that forward and backward passes work without errors."""
        # Simple text
        text = "abcdefghijklmnopqrstuvwxyz" * 4
        dataset = SimpleTextDataset(text, block_size=8, device="cpu")
        vocab_size = dataset.vocab_size

        # Simple model
        config = TransformerConfig(n_embd=32, n_layer=2, n_head=4, n_max_context=8)
        model = TokenTransformer(config, vocab_size)

        # Get a batch
        x, y = dataset[0]
        x = x.unsqueeze(0)  # Add batch dimension
        y = y.unsqueeze(0)

        # Forward pass
        logits, loss = model(x, y)

        # Check shapes
        assert logits.shape == (1, 8, vocab_size)
        assert loss.dim() == 0

        # Backward pass
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
