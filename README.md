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

**Note on ELO variance**: With small test datasets (80 games/gen), ELO results can vary between runs. The `test_model_predictions` test is more reliable for validating that training works correctly.


## Web Application

The project includes a web interface for playing games.
The frontend is written in TypeScript and needs to be compiled.

### Setup & Run
Use the provided script to compile the frontend and start the server:

```bash
./scripts/start_web_app.sh
```

This script will:
1. Install Node.js dependencies (if needed).
2. Compile TypeScript files to `web_app/static/`.
3. Start the FastAPI server with hot-reload enabled.

You can access the game portal at [http://localhost:8000](http://localhost:8000).
To start a new Connect4 game directly: [http://localhost:8000/connect4/new](http://localhost:8000/connect4/new)

# nanoGPT.fork
**nanoGPT.fork** is a fork & wrapper for nanoGPT to make it easier to call from other tools.