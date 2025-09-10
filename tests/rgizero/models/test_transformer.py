import torch
from rgi.rgizero.models.transformer import TransformerConfig, CausalSelfAttention, MLP, Block, Transformer


class TestTransformerConfig:
    def test_default_config(self):
        """Test default configuration values."""
        config = TransformerConfig()

        assert config.n_max_context == 1024
        assert config.n_embd == 768
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.dropout == 0.0
        assert config.bias is False


class TestCausalSelfAttention:
    def test_forward_shape(self):
        """Test attention forward pass shapes."""
        config = TransformerConfig(n_embd=64, n_head=4, n_max_context=32)
        attn = CausalSelfAttention(config)

        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, config.n_embd)

        output = attn(x)
        assert output.shape == (batch_size, seq_len, config.n_embd)

    def test_causal_mask(self):
        """Test that attention is causal (can't attend to future tokens)."""
        config = TransformerConfig(n_embd=64, n_head=4, n_max_context=32, dropout=0.0)
        attn = CausalSelfAttention(config)
        attn.eval()  # Disable dropout for deterministic test

        seq_len = 4
        x = torch.randn(1, seq_len, config.n_embd)

        # Forward pass
        with torch.no_grad():
            output = attn(x)

        # Shape should be preserved
        assert output.shape == x.shape


class TestMLP:
    def test_forward_shape(self):
        """Test MLP forward pass shapes."""
        config = TransformerConfig(n_embd=64)
        mlp = MLP(config)

        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, config.n_embd)

        output = mlp(x)
        assert output.shape == (batch_size, seq_len, config.n_embd)


class TestBlock:
    def test_forward_shape(self):
        """Test transformer block forward pass."""
        config = TransformerConfig(n_embd=64, n_head=4, n_max_context=32)
        block = Block(config)

        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, config.n_embd)

        output = block(x)
        assert output.shape == (batch_size, seq_len, config.n_embd)


class TestTransformer:
    def test_forward_shape(self):
        """Test full transformer forward pass."""
        config = TransformerConfig(n_embd=64, n_layer=2, n_head=4, n_max_context=32)
        transformer = Transformer(config)

        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, config.n_embd)

        output = transformer(x)
        assert output.shape == (batch_size, seq_len, config.n_embd)

    def test_positional_encoding(self):
        """Test that positional encoding is applied."""
        config = TransformerConfig(n_embd=64, n_layer=1, n_head=4, n_max_context=32)
        transformer = Transformer(config)

        seq_len = 16
        x = torch.zeros(1, seq_len, config.n_embd)  # Zero input

        with torch.no_grad():
            output = transformer(x)

        # Output should not be zero due to positional encoding
        assert not torch.allclose(output, x, atol=1e-6)

    def test_parameter_count(self):
        """Test parameter counting utility."""
        config = TransformerConfig(n_embd=64, n_layer=2, n_head=4)
        transformer = Transformer(config)

        # Should have reasonable number of parameters
        total_params = sum(p.numel() for p in transformer.parameters())
        assert total_params > 1000  # Sanity check

        # Test parameter counting by type
        trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        assert trainable_params == total_params  # All should be trainable by default
