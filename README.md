# rgi3

This repository contains (or aims to contain) multiple connected AI tools and experiments.

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


# nanoGPT.fork
**nanoGPT.fork** is a fork & wrapper for nanoGPT to make it easier to call from other tools.
