import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rgi.rgizero.games.connect4 import Connect4Game
from rgi.rgizero.players.random_player import RandomPlayer
from rgi.rgizero.tournament import Tournament

async def main():
    print("Initializing Tournament Demo...")
    
    game = Connect4Game()
    
    # Create some players
    # We'll create a few random players. 
    # To make it interesting, we could try to make a "biased" random player if we had one,
    # but for now just 4 random players to see them draw/win randomly.
    players = {
        "Random_1": lambda: RandomPlayer(game),
        "Random_2": lambda: RandomPlayer(game),
        "Random_3": lambda: RandomPlayer(game),
        "Random_4": lambda: RandomPlayer(game),
    }
    
    tournament = Tournament(game, players, initial_elo=1200)
    
    print(f"Starting tournament with {len(players)} players...")
    print("Running 20 games (concurrently)...")
    
    await tournament.run(num_games=20, concurrent_games=5)
    
    tournament.print_standings()

if __name__ == "__main__":
    asyncio.run(main())
