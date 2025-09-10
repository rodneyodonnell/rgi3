import torch
from unittest.mock import patch
from rgi.rgizero import common


class TestCommon:
    def test_find_project_root(self):
        """Test project root finding."""
        root = common._find_project_root()
        assert root.is_dir()
        assert (root / ".git").exists()
        assert (root / "README.md").exists()

    def test_model_dir_creation(self):
        """Test model directory creation."""
        model_name = "test-model"
        version = "v1"

        model_path = common.model_dir(model_name, version)
        assert model_path.exists()
        assert model_path.is_dir()
        assert model_path.name == version
        assert model_path.parent.name == model_name

    def test_data_dir_creation(self):
        """Test data directory creation."""
        data_name = "test-data"

        data_path = common.data_dir(data_name)
        assert data_path.exists()
        assert data_path.is_dir()
        assert data_path.name == data_name

    def test_get_ctx_cpu(self):
        """Test autocast context for CPU."""
        ctx = common.get_ctx("float32", "cpu")

        # CPU should return nullcontext
        with ctx:
            x = torch.tensor([1.0])
            assert x.dtype == torch.float32

    def test_get_ctx_cuda_types(self):
        """Test autocast context types for CUDA."""
        # Test different dtypes
        for dtype_str, expected_dtype in [
            ("float32", torch.float32),
            ("bfloat16", torch.bfloat16),
            ("float16", torch.float16),
        ]:
            ctx = common.get_ctx(dtype_str, "cuda")

            # Should return autocast context for CUDA
            assert hasattr(ctx, "__enter__")
            assert hasattr(ctx, "__exit__")

    @patch("os.makedirs")
    def test_directories_created_on_import(self, mock_makedirs):
        """Test that directories are created when module is imported."""
        # Reload the module to trigger directory creation
        import importlib

        importlib.reload(common)

        # Should have created DATA_ROOT and MODEL_ROOT
        assert mock_makedirs.call_count >= 2
