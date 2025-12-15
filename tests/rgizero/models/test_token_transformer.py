import torch
from rgi.rgizero.models.transformer import TransformerConfig
from rgi.rgizero.models.token_transformer import TokenTransformer


class TestTokenTransformer:
    def test_initialization(self):
        """Test model initialization."""
        config = TransformerConfig(n_embd=64, n_layer=2, n_head=4, n_max_context=32)
        vocab_size = 100

        model = TokenTransformer(config, vocab_size)

        # Check components exist
        assert hasattr(model, "wte")  # Token embeddings
        assert hasattr(model, "transformer")
        assert hasattr(model, "ln_f")  # Final layer norm
        assert hasattr(model, "lm_head")  # Language model head

        # Check dimensions
        assert model.wte.num_embeddings == vocab_size
        assert model.wte.embedding_dim == config.n_embd
        assert model.lm_head.in_features == config.n_embd
        assert model.lm_head.out_features == vocab_size

    def test_forward_shape(self):
        """Test forward pass shapes."""
        config = TransformerConfig(n_embd=64, n_layer=2, n_head=4, n_max_context=32)
        vocab_size = 100
        model = TokenTransformer(config, vocab_size)

        batch_size, seq_len = 2, 16
        idx = torch.randint(0, vocab_size, (batch_size, seq_len))

        # When no targets provided, model returns only the last position for efficiency
        logits, loss_dict, loss = model(idx)
        assert logits.shape == (batch_size, 1, vocab_size)  # Only last position
        assert loss is None
        assert loss_dict is None

    def test_forward_with_targets(self):
        """Test forward pass with targets (for training)."""
        config = TransformerConfig(n_embd=64, n_layer=2, n_head=4, n_max_context=32)
        vocab_size = 100
        model = TokenTransformer(config, vocab_size)

        batch_size, seq_len = 2, 16
        idx = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits, loss_dict, loss = model(idx, targets)

        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() > 0  # Should have positive loss

    def test_generate(self):
        """Test text generation."""
        config = TransformerConfig(n_embd=64, n_layer=2, n_head=4, n_max_context=32)
        vocab_size = 100
        model = TokenTransformer(config, vocab_size)
        model.eval()

        # Start with single token
        idx = torch.randint(0, vocab_size, (1, 1))
        max_new_tokens = 10

        with torch.no_grad():
            generated = model.generate(idx, max_new_tokens=max_new_tokens)

        assert generated.shape == (1, 1 + max_new_tokens)
        assert torch.all(generated >= 0)
        assert torch.all(generated < vocab_size)

    def test_generate_multiple_sequences(self):
        """Test generating multiple sequences."""
        config = TransformerConfig(n_embd=64, n_layer=2, n_head=4, n_max_context=32)
        vocab_size = 100
        model = TokenTransformer(config, vocab_size)
        model.eval()

        # Start with multiple sequences
        batch_size = 3
        idx = torch.randint(0, vocab_size, (batch_size, 5))
        max_new_tokens = 5

        with torch.no_grad():
            generated = model.generate(idx, max_new_tokens=max_new_tokens)

        assert generated.shape == (batch_size, 5 + max_new_tokens)

    def test_parameter_sharing(self):
        """Test that weight tying works correctly."""
        config = TransformerConfig(n_embd=64, n_layer=2, n_head=4)
        vocab_size = 100
        model = TokenTransformer(config, vocab_size)

        # Check that wte and lm_head share weights (weight tying)
        assert torch.equal(model.wte.weight, model.lm_head.weight)

    def test_context_length_limit(self):
        """Test behavior with sequences longer than context."""
        config = TransformerConfig(n_embd=64, n_layer=2, n_head=4, n_max_context=8)
        vocab_size = 100
        model = TokenTransformer(config, vocab_size)

        # Sequence longer than context
        long_seq = torch.randint(0, vocab_size, (1, 16))

        # Should only use last n_max_context tokens for generation
        with torch.no_grad():
            generated = model.generate(long_seq, max_new_tokens=1)

        # Should have original length + 1
        assert generated.shape == (1, 17)
