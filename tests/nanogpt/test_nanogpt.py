"""Basic unit tests for nanoGPT functionality."""

from rgi.nanogpt import wrapper


class TestNanoGPT:
    """Test cases for nanoGPT wrapper functionality."""

    def test_prepare_shakespeare_data(self):
        """Test that Shakespeare data preparation runs without errors."""
        # This should run the data preparation script
        wrapper.run_nanogpt_script("prepare_shakespeare")
        # If we get here without exception, the test passes

    def test_femto_gpt_training(self):
        """Test femto GPT model training with minimal iterations."""
        # Femto GPT configuration from the notebook
        femto_train_args = [
            "--max_iters=5",  # Very small for testing
            "--eval_interval=2",
            "--eval_iters=1",
            "--log_interval=1",
            "--batch_size=16",
            "--block_size=128",
            "--n_layer=4",
            "--n_head=4",
            "--n_embd=128",
            "--dropout=0.2",
            "--warmup_iters=2",
            "--compile=False",  # Disable compilation for faster testing
        ]

        # Run training with femto GPT config
        wrapper.train(config_file="train_shakespeare_char.py", argv=femto_train_args)
        # If we get here without exception, the test passes

    def test_sample_generation(self):
        """Test sample generation with minimal training."""
        # First do minimal training
        train_args = [
            "--max_iters=5",
            "--batch_size=16",
            "--block_size=128",
            "--compile=False",
        ]

        # Train a tiny model
        wrapper.train(config_file="train_shakespeare_char.py", argv=train_args)

        # Then generate samples
        wrapper.sample_data(config_file="train_shakespeare_char.py", argv=train_args)
        # If we get here without exception, the test passes
