# RGI project repo

This is an experimental repository for building RL & LLM tools.

# rgizero

**rgizero** is a general-purpose AlphaZero-style game player, with goals of supporting multimodal capabilities.

The goal is to create a reinforcement learning agent that can:
1.  Initially master various board games (Connect4, Othello, etc.) using a unified MCTS + Transformer architecture.
2.  Be extended to games which are not perfect-information or fully deterministic.
3.  Experiment with "multimodal" and RL ideas so the same model architecture could potentially both play games and have "chatbot" capabilities.

## Quick Start

### Installation

```bash
uv sync
```

### Playing a Game

TODO: Add notebook showing this.

## Supported Games

-   Connect4
-   Count21 (Simple counting game)
-   Othello (Reversi)

## Testing

### Running Tests

Run all tests:
```bash
uv run pytest
```

Run unit tests only (fast):
```bash
uv run pytest tests/ -m "not integration"
```

Run integration tests (full end-to-end pipeline, ~5-15 minutes):
```bash
uv run pytest tests/rgizero/test_integration.py -v
```

### Integration Tests

The integration tests in `tests/rgizero/test_integration.py` validate the full AlphaZero training pipeline:

- **`test_full_training_pipeline_count21`** - Basic pipeline test (3 generations, ~5 seconds)
- **`test_full_training_pipeline_connect4`** - Connect4 training (3 generations, ~2-3 minutes)
- **`test_model_improvement_validation`** - Validates trained model beats random baseline via ELO (~2-3 minutes)
- **`test_elo_progression_across_generations`** - Validates ELO improvement trend across 4 generations (~3-4 minutes)
- **`test_experiment_forking`** - Tests experiment forking/continuation (~2-3 minutes)

These tests ensure you can make changes to the codebase without breaking the core training loop.

Run a single test:
```bash
uv run pytest tests/rgizero/test_integration.py::test_full_training_pipeline_count21 -v -s
```


# nanoGPT.fork
**nanoGPT.fork** is a fork & wrapper for nanoGPT to make it easier to call from other tools.