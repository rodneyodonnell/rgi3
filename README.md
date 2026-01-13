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

## Training Models

Train AlphaZero-style models for N generations with automatic ELO evaluation to track improvement.

### Quick Training Run

Train a model across multiple generations and see ELO progression:

```bash
# Train Connect4 for 20 generations
uv run python scripts/train_long_experiment.py --game connect4 --generations 20 --games-per-gen 150

# Train Othello for 15 generations
uv run python scripts/train_long_experiment.py --game othello --generations 15 --games-per-gen 200

# Quick test with small config
uv run python scripts/train_long_experiment.py --game count21 --generations 10 --games-per-gen 100
```

**What it does:**
- Trains for N generations using self-play
- Saves all generation models to `experiments/long_run/<game>/models/`
- Runs round-robin ELO tournament at the end
- Evaluates every 5th generation + final generation (configurable with `--tournament-interval`)
- Shows ELO progression from Gen 0 (random baseline) to final model
- Validates that training is working by comparing against random play

### Training Scripts

The repository includes several training scripts:

1. **`scripts/train_long_experiment.py`** - Main training script with ELO evaluation
   - Supports: `count21`, `connect4`, `othello`
   - Configurable generations, games per generation, MCTS simulations
   - Full ELO tournament at end with progression analysis

2. **`scripts/train_connect4_60min.py`** - Time-limited Connect4 training
   - Stops after specified time limit (default: 60 minutes)
   - Periodic ELO checks during training (every N generations)
   - Useful for quick experiments with time constraints

3. **`scripts/overnight_experiments.py`** - Multi-variant hyperparameter search
   - Runs multiple experiment configurations automatically
   - Compares different model sizes, learning rates, etc.
   - Maintains incremental ELO rankings across all variants

### Example Output

After training completes, you'll see ELO progression analysis:

```
ELO PROGRESSION ANALYSIS
================================================================================
Gen  0: ELO= 1000.0, Games= 40, Wins= 12 (RANDOM BASELINE)
Gen  5: ELO= 1089.3, Games= 40, Wins= 18
Gen 10: ELO= 1156.7, Games= 40, Wins= 23
Gen 15: ELO= 1203.2, Games= 40, Wins= 27 ⭐ BEST MODEL

Improvement from Gen 0 to Gen 15: +203.2 ELO
✓ Final model beats random baseline by 203.2 ELO
```

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