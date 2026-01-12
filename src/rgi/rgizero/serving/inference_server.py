"""
Inference server for gRPC-based model serving.

Uses pipelined parser/GPU threads and bytes proto fields for optimal performance.
"""

import queue
import threading
import time
from typing import Optional

import grpc
import numpy as np
import torch
from concurrent import futures

from rgi.rgizero.serving import inference_pb2, inference_pb2_grpc
from rgi.rgizero.evaluators import ActionHistoryTransformerEvaluator
from rgi.rgizero.models.action_history_transformer import ActionHistoryTransformer, TransformerConfig


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """gRPC servicer with pipelined CPU/GPU processing."""

    def __init__(
        self,
        evaluator: ActionHistoryTransformerEvaluator,
        vocab_size: int,
        num_players: int,
        verbose: bool = False,
    ):
        self.evaluator = evaluator
        self.vocab_size = vocab_size
        self.num_players = num_players
        self.verbose = verbose

        # Request queue (from gRPC handlers)
        self.request_queue: queue.Queue = queue.Queue()
        # Parsed batch queue (parser → GPU)
        self.parsed_queue: queue.Queue = queue.Queue(maxsize=4)

        self.stop_event = threading.Event()

        self.total_batches = 0
        self.total_evals = 0
        self.start_time = time.time()

        # Start pipeline threads
        self.parser_thread = threading.Thread(target=self._parser_loop, daemon=True)
        self.gpu_thread = threading.Thread(target=self._gpu_loop, daemon=True)
        self.parser_thread.start()
        self.gpu_thread.start()

    def _parser_loop(self):
        """Collects requests and prepares numpy arrays for GPU."""
        while not self.stop_event.is_set():
            pending = []

            try:
                req = self.request_queue.get(timeout=0.01)
                pending.append(req)
            except queue.Empty:
                continue

            # Grab all available
            while not self.request_queue.empty() and len(pending) < 10000:
                req = self.request_queue.get()
                pending.append(req)

            total_B = sum(r[0].batch_size for r in pending)
            max_len = max(r[0].max_len for r in pending)

            # Pre-allocate combined arrays
            x_combined = np.zeros((total_B, max_len), dtype=np.int32)
            len_combined = np.zeros(total_B, dtype=np.int32)
            mask_combined = np.zeros((total_B, self.vocab_size), dtype=np.bool_)

            idx = 0
            bounds = []
            for request, event, response_holder in pending:
                B = request.batch_size
                req_max_len = request.max_len

                # Zero-copy: use np.frombuffer on bytes
                x_req = np.frombuffer(request.x_data_bytes, dtype=np.int32).reshape(B, req_max_len)
                x_combined[idx : idx + B, :req_max_len] = x_req
                len_combined[idx : idx + B] = np.frombuffer(request.encoded_lengths_bytes, dtype=np.int32)

                mask_req = np.frombuffer(request.legal_mask_bytes, dtype=np.bool_).reshape(B, self.vocab_size)
                mask_combined[idx : idx + B] = mask_req

                bounds.append((idx, B, request, event, response_holder))
                idx += B

            legal_indices = [np.where(mask_combined[i])[0] for i in range(total_B)]

            # Put parsed batch on GPU queue
            parsed_batch = (x_combined, len_combined, mask_combined, legal_indices, bounds, total_B)
            self.parsed_queue.put(parsed_batch)

    def _gpu_loop(self):
        """Runs GPU inference on parsed batches."""
        while not self.stop_event.is_set():
            try:
                parsed_batch = self.parsed_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            x_combined, len_combined, mask_combined, legal_indices, bounds, total_B = parsed_batch

            results = self.evaluator.infer_from_encoded(x_combined, len_combined, mask_combined, legal_indices)

            # Build responses with bytes for zero-copy
            for start_idx, count, request, event, response_holder in bounds:
                worker_results = results[start_idx : start_idx + count]

                # Concatenate all policies and values
                all_policies = np.concatenate([r.legal_policy for r in worker_results]).astype(np.float32)
                all_values = np.concatenate([r.player_values for r in worker_results]).astype(np.float32)

                response = inference_pb2.EncodedEvalResponse(
                    worker_id=request.worker_id,
                    request_id=request.request_id,
                    num_players=self.num_players,
                    legal_policies_bytes=all_policies.tobytes(),
                    player_values_bytes=all_values.tobytes(),
                    total_legal_actions=len(all_policies),
                )

                response_holder["response"] = response
                event.set()

            self.total_batches += 1
            self.total_evals += total_B

            if self.verbose and self.total_batches % 100 == 0:
                elapsed = time.time() - self.start_time
                print(
                    f"InferenceServer: batches={self.total_batches}, evals={self.total_evals}, "
                    f"evals/sec={self.total_evals / elapsed:.0f}"
                )

    def EvaluateEncoded(self, request, context):
        event = threading.Event()
        response_holder: dict = {}
        self.request_queue.put((request, event, response_holder))
        event.wait(timeout=30)
        return response_holder.get("response", inference_pb2.EncodedEvalResponse())

    def stop(self):
        self.stop_event.set()
        self.parser_thread.join(timeout=2)
        self.gpu_thread.join(timeout=2)


class InferenceServer:
    """Wrapper that manages a gRPC server for model inference."""

    def __init__(
        self,
        evaluator: ActionHistoryTransformerEvaluator,
        vocab_size: int,
        num_players: int,
        port: int = 50051,
        max_workers: int = 50,
        verbose: bool = False,
    ):
        self.evaluator = evaluator
        self.vocab_size = vocab_size
        self.num_players = num_players
        self.port = port
        self.max_workers = max_workers
        self.verbose = verbose

        self.server: Optional[grpc.Server] = None
        self.servicer: Optional[InferenceServicer] = None

    def start(self):
        """Start the gRPC server."""
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=self.max_workers))
        self.servicer = InferenceServicer(
            evaluator=self.evaluator,
            vocab_size=self.vocab_size,
            num_players=self.num_players,
            verbose=self.verbose,
        )
        inference_pb2_grpc.add_InferenceServiceServicer_to_server(self.servicer, self.server)
        self.server.add_insecure_port(f"[::]:{self.port}")
        self.server.start()

        if self.verbose:
            print(f"InferenceServer started on port {self.port}")

    def stop(self, grace: float = 1.0):
        """Stop the gRPC server."""
        if self.servicer:
            self.servicer.stop()
        if self.server:
            self.server.stop(grace=grace)

        if self.verbose:
            print(f"InferenceServer stopped (port {self.port})")

    def wait_for_termination(self, timeout: Optional[float] = None):
        """Block until server terminates."""
        if self.server:
            self.server.wait_for_termination(timeout=timeout)


def run_server_process(
    model_path: str,
    game_name: str,
    port: int,
    ready_event,
    stop_event,
    verbose: bool = False,
):
    """Entry point for running inference server in a subprocess.

    Args:
        model_path: Path to saved model checkpoint
        game_name: Name of game (e.g., "othello")
        port: Port to listen on
        ready_event: multiprocessing.Event to signal when ready
        stop_event: multiprocessing.Event to signal when to stop
        verbose: Whether to print status updates
    """
    from rgi.rgizero.games import game_registry
    from rgi.rgizero.data.trajectory_dataset import Vocab
    from rgi.rgizero.common import TOKENS

    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    # Load model from checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Check if we have metadata in checkpoint
    if isinstance(checkpoint, dict):
        if "vocab" in checkpoint:
            vocab = Vocab.from_dict(checkpoint["vocab"])
        else:
            game = game_registry.create_game(game_name)
            vocab = Vocab(itos=[TOKENS.START_OF_GAME] + list(game.base_game.all_actions()))

        if "num_players" in checkpoint:
            num_players = checkpoint["num_players"]
        else:
            if "game" not in locals():
                game = game_registry.create_game(game_name)
            num_players = game.num_players(game.initial_state())
    else:
        # Fallback for old/direct model saves
        game = game_registry.create_game(game_name)
        vocab = Vocab(itos=[TOKENS.START_OF_GAME] + list(game.base_game.all_actions()))
        num_players = game.num_players(game.initial_state())

    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        # Re-instantiate from config and state dict
        config = TransformerConfig(**checkpoint["model_config"])
        model = ActionHistoryTransformer(config=config, action_vocab_size=vocab.vocab_size, num_players=num_players)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            # Fallback if just config provided (unlikely) or flat dict
            try:
                model.load_state_dict(checkpoint)
            except Exception:
                pass
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        # Maybe model object is inside 'model' key (legacy?)
        model = checkpoint["model"]
    else:
        # Assume it's the model object itself
        model = checkpoint

    model = model.to(device)
    model.eval()

    evaluator = ActionHistoryTransformerEvaluator(
        model=model,
        device=device,
        block_size=100,
        vocab=vocab,
        verbose=False,
    )

    # Start server
    server = InferenceServer(
        evaluator=evaluator,
        vocab_size=vocab.vocab_size,
        num_players=num_players,
        port=port,
        verbose=verbose,
    )
    server.start()

    # Signal ready
    ready_event.set()

    # Wait for stop signal
    while not stop_event.is_set():
        time.sleep(0.1)

    server.stop()
