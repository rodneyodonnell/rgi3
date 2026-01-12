"""
Model server manager for lazy startup of inference servers.

Manages inference server processes on-demand.
"""

import multiprocessing as mp
from pathlib import Path
from typing import Dict, Tuple

from rgi.rgizero.serving.inference_server import run_server_process


class ModelServerManager:
    """Manages inference server processes on-demand.

    Starts servers lazily when first requested, keeps them running
    for the duration of use.
    """

    def __init__(
        self,
        base_port: int = 50051,
        startup_timeout: float = 30.0,
        verbose: bool = False,
    ):
        self.base_port = base_port
        self.startup_timeout = startup_timeout
        self.verbose = verbose

        self.servers: Dict[str, Tuple[int, mp.Process, mp.Event]] = {}  # model_path → (port, process, stop_event)
        self.next_port = base_port

    def get_port(self, model_path: str, game_name: str) -> int:
        """Get the port for a model. Starts server if not running.

        Args:
            model_path: Path to model checkpoint
            game_name: Name of game (e.g., "othello")

        Returns:
            Port number for the inference server
        """
        model_path = str(Path(model_path).resolve())

        if model_path in self.servers:
            port, process, _ = self.servers[model_path]
            if process.is_alive():
                return port
            else:
                # Process died, clean up and restart
                del self.servers[model_path]

        # Start new server
        port = self._start_server(model_path, game_name)
        return port

    def _start_server(self, model_path: str, game_name: str) -> int:
        """Start a new inference server process."""
        port = self.next_port
        self.next_port += 1

        ready_event = mp.Event()
        stop_event = mp.Event()

        process = mp.Process(
            target=run_server_process,
            args=(model_path, game_name, port, ready_event, stop_event, self.verbose),
        )
        process.start()

        # Wait for server to be ready
        if not ready_event.wait(timeout=self.startup_timeout):
            process.terminate()
            raise RuntimeError(f"Inference server failed to start within {self.startup_timeout}s")

        self.servers[model_path] = (port, process, stop_event)

        if self.verbose:
            print(f"Started inference server for {Path(model_path).name} on port {port}")

        return port

    def stop_server(self, model_path: str):
        """Stop a specific server."""
        model_path = str(Path(model_path).resolve())

        if model_path in self.servers:
            port, process, stop_event = self.servers[model_path]
            stop_event.set()
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
            del self.servers[model_path]

    def shutdown_all(self):
        """Stop all running servers."""
        for model_path in list(self.servers.keys()):
            self.stop_server(model_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown_all()
        return False
