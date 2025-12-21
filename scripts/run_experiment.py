
import asyncio
import argparse
from pathlib import Path
from rgi.rgizero.experiment import ExperimentConfig, ExperimentRunner

async def main():
    parser = argparse.ArgumentParser(description="Run RGI Zero Experiment")
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--game", type=str, default="connect4", help="Game name")
    parser.add_argument("--gens", type=int, default=10, help="Number of generations")
    parser.add_argument("--games-per-gen", type=int, default=1000, help="Games per generation")
    parser.add_argument("--sims", type=int, default=100, help="MCTS simulations")
    parser.add_argument("--size", type=str, default="tiny", help="Model size (tiny, small, large)")
    parser.add_argument("--parent", type=str, default=None, help="Parent experiment for forking")
    parser.add_argument("--parent-cap", type=int, default=None, help="Max generation to use from parent")
    
    args = parser.parse_args()
    
    base_dir = Path("data/experiments")
    
    config = ExperimentConfig(
        experiment_name=args.name,
        game_name=args.game,
        num_generations=args.gens,
        num_games_per_gen=args.games_per_gen,
        num_simulations=args.sims,
        model_size=args.size,
        parent_experiment_name=args.parent,
        parent_generation_cap=args.parent_cap
    )
    
    runner = ExperimentRunner(config, base_dir)
    await runner.run()

if __name__ == "__main__":
    asyncio.run(main())
