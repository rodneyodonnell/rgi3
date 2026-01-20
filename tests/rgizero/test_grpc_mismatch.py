import numpy as np
from unittest.mock import MagicMock
import torch

from rgi.rgizero.data.trajectory_dataset import Vocab
from rgi.rgizero.evaluators import ActionHistoryTransformerEvaluator


class TestGrpcMismatch:
    """Test suite for policy ordering and duplicate handling (gRPC regression)."""

    def test_evaluator_preserves_order_and_duplicates(self):
        """Verify that infer_from_encoded respects the order and multiplicity of legal_indices.

        Old behavior: used boolean masking, which sorted indices and removed duplicates.
        New behavior: uses advanced indexing, preserving order and duplicates.
        """
        # Setup
        vocab_size = 20
        B = 1

        # indices: [10, 5, 10] (Unsorted, Duplicates)
        legal_indices = [np.array([10, 5, 10], dtype=np.int32)]

        # Mask would simulate what the old code/server did (collapsing to [5, 10])
        legal_mask = np.zeros((B, vocab_size), dtype=np.bool_)
        legal_mask[0, [10, 5, 10]] = True

        x_np = np.zeros((B, 5), dtype=np.int32)
        encoded_len_np = np.array([5], dtype=np.int32)

        # Mock Model
        # We want to return specific logits to trace them.
        # Logit for 5 -> 100
        # Logit for 10 -> 200
        # Others -> -inf

        # We can mock the model output directly or just check the evaluator logic.
        # Since evaluator calls model, we need a flexible model or mock.

        class MockModel(torch.nn.Module):
            def __call__(self, x, encoded_len=None):
                B, T = x.shape
                # Output shape: (B, 1, Vocab), (B, 1, Players)
                policy_logits = torch.zeros((B, 1, vocab_size))

                # Set distinctive logits
                policy_logits[0, 0, 5] = 10.0
                policy_logits[0, 0, 10] = 20.0

                value_logits = torch.zeros((B, 1, 2))
                return (policy_logits, value_logits), None, None

            def eval(self):
                return self

        model = MockModel()
        evaluator = ActionHistoryTransformerEvaluator(
            model=model,  # type: ignore
            device="cpu",  # Test on CPU
            block_size=10,
            vocab=MagicMock(spec=Vocab),
            verbose=False,
        )
        # Hack: vocab mock needed? Not for infer_from_encoded.
        evaluator.vocab.stoi = {str(i): i for i in range(vocab_size)}  # type: ignore

        # Run Inference
        results = evaluator.infer_from_encoded(x_np, encoded_len_np, legal_mask, legal_indices)
        result = results[0]

        # Verify
        # Expected: [Prob(10), Prob(5), Prob(10)]
        # Logits: 5->10, 10->20.
        # Prob ~ exp(logit). exp(20) >> exp(10).
        # Softmax over masked values only?
        # The evaluator masks invalid actions primarily.

        legal_policy = result.legal_policy

        # Check Length (Fixes AssertionError)
        assert len(legal_policy) == 3, f"Expected length 3 (duplicates preserved), got {len(legal_policy)}"

        # Check Ordering
        # policy[0] should be prob for action 10 (high)
        # policy[1] should be prob for action 5 (low)
        # policy[2] should be prob for action 10 (high)

        assert legal_policy[0] > legal_policy[1], "First element (10) should be > Second (5)"
        assert np.isclose(legal_policy[0], legal_policy[2]), "First (10) and Third (10) should be identical"

        print("Evaluator correctly preserved order and duplicates.")

    def test_worker_remapping_logic(self):
        """Verify the client-side remapping logic used in selfplay_worker.py.

        Simulates:
        1. Client has 'encoded_actions' (original unsorted/duplicates).
        2. Server returns 'all_policies' corresponding to sorted unique mask.
        3. Client reconstructs 'legal_policy'.
        """

        # Scenario:
        # Original: [10, 5, 10]
        encoded_actions = np.array([10, 5, 10], dtype=np.int64)

        # Server Logic (Simulation): masking and sorting
        unique_indices = np.unique(encoded_actions)  # [5, 10]

        # Server Policy (Simulated Values)
        # Let's say Prob(5)=0.2, Prob(10)=0.8
        # Server returns values corresponding to [5, 10]
        server_policy_subset = np.array([0.2, 0.8], dtype=np.float32)

        # --- Client Patch Logic Start ---

        # 2. Map sorted unique indices to the values
        vocab_to_value = {idx: val for idx, val in zip(unique_indices, server_policy_subset)}

        # 3. Reconstruct full ordered policy
        reconstructed_policy = np.array([vocab_to_value[idx] for idx in encoded_actions], dtype=np.float32)

        # --- Client Patch Logic End ---

        # Assertions
        assert len(reconstructed_policy) == 3
        # Should be [0.8, 0.2, 0.8]
        expected = np.array([0.8, 0.2, 0.8], dtype=np.float32)
        np.testing.assert_array_almost_equal(reconstructed_policy, expected)

        print("Worker remapping logic correctly reconstructed policy.")


if __name__ == "__main__":
    t = TestGrpcMismatch()
    t.test_evaluator_preserves_order_and_duplicates()
    t.test_worker_remapping_logic()
