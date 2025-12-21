import torch


def validate_probabilities_or_die(tensor: torch.Tensor, dim: int = 1, tol: float = 1e-6) -> bool:
    """Validates that a tensor contains valid probabilities.

    Args:
        tensor: The tensor to validate.
        dim: The dimension to sum over.
        tol: Tolerance for sum checking.

    Returns:
        True if valid.

    Raises:
        ValueError: If values are not in [0, 1] or do not sum to 1.0.
    """
    # 1. Check if all values are >= 0 and <= 1
    in_range = (tensor >= 0).all() and (tensor <= 1).all()
    if not in_range:
        raise ValueError(f"Probabilities are not in range [0, 1]: {tensor}")

    # 2. Check if sum is approximately 1.0
    row_sums = tensor.sum(dim=dim)
    sums_to_one = torch.allclose(row_sums, torch.ones_like(row_sums), atol=tol)
    if not sums_to_one:
        raise ValueError(f"Probabilities do not sum to 1.0: {tensor}, sums: {row_sums}")

    return bool(in_range) and bool(sums_to_one)
