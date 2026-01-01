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