# rgizero

**rgizero** is a general-purpose AlphaZero implementation designed towards multimodal capabilities.

The goal is to create a reinforcement learning agent that can:
1.  Master various board games (Connect4, Othello, etc.) using a unified MCTS + Transformer architecture.
2.  Be "multimodal" in the sense that the same model architecture could potentially ingest rulebooks (text) or board states (images/tokens) and learn to play effectively.

## Quick Start

### Installation

Ensure you have the dependencies installed (using `uv` or similar):

```bash
uv sync
```

### Playing a Game

You can easily instantiate games using the registry:

```python
from rgi.rgizero.games.game_registry import create_game

# Create a Connect4 game
game = create_game("connect4")
state = game.initial_state()
print(game.pretty_str(state))
```

### Running a Tournament

To run a demo tournament between random players:

```bash
python scripts/run_tournament_demo.py
```

## Architecture

-   **MCTS**: AlphaZero-style Monte Carlo Tree Search.
-   **Model**: Transformer-based architecture (`ActionHistoryTransformer`) acting as both Policy and Value network.
-   **Training**: Iterative self-play and training loop (currently being extracted from notebooks).

## Features

-   **Game Registry**: Easy access to supported games.
-   **Async MCTS**: High-throughput game playing using `asyncio`.
-   **Hyperparameter Tuning**: Built-in tuner for optimizing model parameters.
-   **Tournament System**: ELO-based evaluation of different agents.

## Supported Games

-   Connect4
-   Count21 (Simple counting game)
-   Othello (Reversi)

## Future Roadmap

-   **Performance**: Batched inference for massive speedups.
-   **Multimodal**: Integration of text/image encoders.
-   **New Games**: Power Grid and other complex strategy games.