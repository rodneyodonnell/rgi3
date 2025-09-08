"""Wrapper functions for calling fork of nanoGPT."""

import os
import sys
from contextlib import contextmanager

import pathlib

# Base directory of 'fork' module.
_FORK_BASE_PATH = pathlib.Path(os.path.dirname(__file__)) / "fork"


def import_or_reload(module_name):
    """reload module if it exists, otherwise import it. return the module

    This approach is useful for running 'train.py' and other scripts which are not
    designed to be run as libraries. We can rerun them with different sys.argv, etc.
    """
    import importlib

    module = sys.modules.get(module_name)
    if module:
        importlib.reload(module)
    else:
        module = importlib.import_module(module_name)
    return module


@contextmanager
def custom_argv(argv):
    """Temporarily override sys.argv to run 'scripts' as 'libraries'"""
    old_argv = sys.argv
    sys.argv = argv
    yield
    sys.argv = old_argv


def run_nanogpt_script(script_name: str, argv: list[str] | None = None):
    """
    Run a nanoGPT script with the given arguments.

    Sample usage:
    - wrapper.run_nanogpt_script("prepare_shakespeare", args=[])
    - wrapper.run_nanogpt_script("train", args=["config_file.py", "--max_iters=100"])
    """
    with custom_argv(argv or []):
        import_or_reload(f"rgi.nanogpt.fork.{script_name}")


def train(config_file: str, argv: list[str]):
    config_path = str(_FORK_BASE_PATH / "config" / config_file)
    argv = ["train.py", config_path] + argv
    run_nanogpt_script("train", argv)


def sample_data(config_file: str, argv: list[str]):
    config_path = str(_FORK_BASE_PATH / "config" / config_file)
    argv = ["sample.py", config_path] + argv
    run_nanogpt_script("sample", argv)
