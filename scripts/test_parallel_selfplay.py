#!/usr/bin/env python3
"""
Test script for hybrid parallel self-play with async workers.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from rgi.rgizero.common import TOKENS
from rgi.rgizero.data.trajectory_dataset import Vocab
from rgi.rgizero.games import game_registry
from rgi.rgizero.models.transformer import TransformerConfig
from rgi.rgizero.models.tuner import create_random_model
from rgi.rgizero.inference_server import HybridParallelSelfPlay


def main():
    game_name = "othello"
    num_games = 40
    num_workers = 4
    concurrent_games_per_worker = 50  # 50 * 4 = 200 concurrent games (match async)
    num_simulations = 30
    
    print(f"Testing HYBRID parallel self-play")
    print(f"Game: {game_name}")
    print(f"Workers: {num_workers}")
    print(f"Concurrent games per worker: {concurrent_games_per_worker}")
    print(f"Total concurrent: {num_workers * concurrent_games_per_worker}")
    print(f"Total games: {num_games}, Simulations: {num_simulations}")
    
    # Setup
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    game = game_registry.create_game(game_name)
    vocab = Vocab(itos=[TOKENS.START_OF_GAME] + list(game.base_game.all_actions()))
    num_players = game.num_players(game.initial_state())
    
    # Create random model
    model_config = TransformerConfig(
        n_max_context=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
    )
    
    model = create_random_model(
        model_config,
        vocab.vocab_size,
        num_players,
        seed=42,
        device=device,
    )
    
    # Create hybrid parallel self-play
    hybrid_selfplay = HybridParallelSelfPlay(
        model=model,
        game_name=game_name,
        vocab=vocab,
        device=device,
        num_workers=num_workers,
        concurrent_games_per_worker=concurrent_games_per_worker,
        num_simulations=num_simulations,
        verbose=True,
    )
    
    # Run
    print("\nStarting hybrid parallel self-play...")
    start_time = time.time()
    
    trajectories = hybrid_selfplay.play_games(num_games, seed=42)
    
    duration = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Games played: {len(trajectories)}")
    print(f"Duration: {duration:.2f}s")
    print(f"Games/sec: {len(trajectories) / duration:.2f}")
    
    # Verify trajectories are valid
    winners = [t["winner"] for t in trajectories]
    print(f"Winners distribution: Player1={winners.count(1)}, Player2={winners.count(2)}, Draw={winners.count(None)}")
    
    avg_moves = sum(len(t["action_history"]) for t in trajectories) / len(trajectories)
    print(f"Average moves per game: {avg_moves:.1f}")


if __name__ == "__main__":
    main()

