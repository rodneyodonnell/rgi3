# AlphaZero Training Guide

This guide explains how to train AlphaZero models for new games in this codebase.

## Overview

The training pipeline follows the standard AlphaZero approach:
1.  **Self-Play**: The current model plays games against itself using MCTS to generate training data (trajectories).
2.  **Training**: The model is trained on these trajectories to better predict MCTS outcomes (policy and value).
3.  **Iteration**: The process repeats, with each generation ideally becoming stronger than the previous one.

## Core Components

- **`Game` Implementation**: The game logic (rules, state, legal actions).
- **`ExperimentConfig`**: Defines the experiment details (game name, generations, games per gen, simulations).
- **`ExperimentRunner`**: orchestrates the self-play and training loop.
- **`AlphazeroPlayer`**: The AI player that uses MCTS and a neural network evaluator.
- **`Trainer`**: Handles the standard supervised training of the model on collected data.

## Steps to Train a New Game

### 1. Register the Game
Ensure your game is implemented in `src/rgi/rgizero/games/` and registered in `game_registry.py`.

### 2. Configure the Experiment
Create an `ExperimentConfig` and an `ExperimentRunner`. You will need to provide `training_args` for the Transformer model.

### 3. Run Training
You can use a general script like `scripts/train_long_experiment.py` or a specialized one for your game.

Example command:
```bash
python scripts/train_long_experiment.py --game connect4 --generations 50 --games-per-gen 100 --simulations 50
```

### 4. Evaluate Progress
Use the `Tournament` class to run models from different generations against each other or against a random baseline.

## Best Practices
- **Simulations**: More simulations during self-play lead to better data but take longer. 50-100 is a good starting point for fast iteration.
- **Micro-batches**: Use `gradient_accumulation_steps` to simulate larger batch sizes if memory is limited.
- **ELO Tracking**: Periodically run tournaments against Gen 0 to ensure the model is actually improving.
