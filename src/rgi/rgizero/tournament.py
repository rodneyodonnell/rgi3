import asyncio
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm.asyncio import tqdm

from rgi.rgizero.games.base import Game
from rgi.rgizero.players.base import Player
from rgi.rgizero.players.alphazero import play_game_async


@dataclass
class PlayerStats:
    elo: float
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    history: List[float] = field(default_factory=list)  # History of ELO ratings


class Tournament:
    def __init__(
        self,
        game: Game,
        players: Dict[str, Player],
        initial_elo: float = 1200.0,
        k_factor: float = 32.0,
    ):
        self.game = game
        self.players = players
        self.k_factor = k_factor
        self.stats: Dict[str, PlayerStats] = {
            pid: PlayerStats(elo=initial_elo, history=[initial_elo]) for pid in players
        }
        self.match_history: List[Tuple[str, str, float]] = [] # (p1, p2, result for p1)

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))

    def update_elo(self, player_a: str, player_b: str, score_a: float):
        """Update ELO ratings after a game.
        
        score_a: 1.0 for win, 0.5 for draw, 0.0 for loss.
        """
        rating_a = self.stats[player_a].elo
        rating_b = self.stats[player_b].elo

        expected_a = self._expected_score(rating_a, rating_b)
        expected_b = self._expected_score(rating_b, rating_a)
        
        score_b = 1.0 - score_a

        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (score_b - expected_b)

        self.stats[player_a].elo = new_rating_a
        self.stats[player_b].elo = new_rating_b
        
        self.stats[player_a].history.append(new_rating_a)
        self.stats[player_b].history.append(new_rating_b)
        
        self.stats[player_a].games_played += 1
        self.stats[player_b].games_played += 1
        
        if score_a == 1.0:
            self.stats[player_a].wins += 1
            self.stats[player_b].losses += 1
        elif score_a == 0.0:
            self.stats[player_a].losses += 1
            self.stats[player_b].wins += 1
        else:
            self.stats[player_a].draws += 1
            self.stats[player_b].draws += 1
            
        self.match_history.append((player_a, player_b, score_a))

    def matchmake(self, num_games: int) -> List[Tuple[str, str]]:
        """Generate pairings for the next round of games.
        
        Strategy:
        1. Prioritize pairs that have played fewer games against each other.
        2. Prioritize pairs with similar ELO ratings.
        """
        player_ids = list(self.players.keys())
        if len(player_ids) < 2:
            return []
            
        # Count games between pairs
        pair_counts = {}
        for p1, p2, _ in self.match_history:
            pair = tuple(sorted((p1, p2)))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            
        pairings = []
        
        # Simple approach: generate random candidates and score them
        # We want to generate 'num_games' pairings.
        # Since we run games in parallel, we can just return a list of pairs.
        # It's okay if a player plays multiple games in a batch if the runner supports it,
        # but for simplicity let's assume we just want to generate a batch of games.
        
        # Let's try to pick pairs that minimize:
        # score = w1 * |elo_diff| + w2 * games_played
        
        # To avoid being too deterministic, we can sample.
        
        for _ in range(num_games):
            # Pick two distinct players
            # We'll use a weighted random choice to pick the first player, maybe favoring those with fewer games?
            # For now, uniform random is fine.
            p1 = random.choice(player_ids)
            
            # Pick p2 based on score
            candidates = [p for p in player_ids if p != p1]
            if not candidates:
                continue
                
            best_p2 = None
            best_score = float('inf')
            
            # Shuffle candidates to break ties randomly
            random.shuffle(candidates)
            
            for p2 in candidates:
                elo_diff = abs(self.stats[p1].elo - self.stats[p2].elo)
                pair = tuple(sorted((p1, p2)))
                played = pair_counts.get(pair, 0)
                
                # We want small elo diff and small played count
                # Normalize elo diff roughly (say 400 points is "large")
                # Normalize played count (say 10 games is "large")
                
                score = (elo_diff / 400.0) + (played * 0.5) 
                
                # Add some noise to explore
                score += random.random() * 0.5
                
                if score < best_score:
                    best_score = score
                    best_p2 = p2
            
            if best_p2:
                pairings.append((p1, best_p2))
                # Update pair count tentatively so we don't pick the same pair too much in one batch
                pair = tuple(sorted((p1, best_p2)))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        return pairings

    async def run(self, num_games: int, concurrent_games: int = 10):
        """Run the tournament."""
        
        # We'll generate games in batches or just a stream.
        # Since we want to update ELO after games to inform matchmaking, 
        # doing it in smaller batches or a loop is better.
        
        # However, to maximize parallelism, we might want a large batch.
        # But if we do a large batch, matchmaking won't adapt.
        # Compromise: batch size = concurrent_games * 2
        
        games_remaining = num_games
        pbar = tqdm(total=num_games, desc="Tournament Progress")
        
        while games_remaining > 0:
            batch_size = min(games_remaining, concurrent_games * 2)
            pairings = self.matchmake(batch_size)
            
            tasks = []
            for p1_id, p2_id in pairings:
                tasks.append(self._play_single_game(p1_id, p2_id))
            
            # Run batch
            results = await asyncio.gather(*tasks)
            
            # Update stats (already done in _play_single_game, but let's verify)
            # Actually _play_single_game calls update_elo, so we are good.
            
            games_remaining -= len(results)
            pbar.update(len(results))
            
        pbar.close()
        
    async def _play_single_game(self, p1_id: str, p2_id: str):
        player1 = self.players[p1_id]
        player2 = self.players[p2_id]
        
        # play_game_async expects a list of players
        # We need to be careful if players are stateful or not reusable concurrently.
        # The AlphaZero players might share a network evaluator which handles batching.
        # If the player objects themselves track state during a game, we might need to clone them or factory them.
        # Assuming Player objects are stateless regarding the game progress (they take game_state as input),
        # but they might have internal MCTS trees.
        # AlphaZeroPlayer has an IncrementalTreeCache. If we reuse the same player instance in parallel games,
        # the cache will be clobbered.
        # So we SHOULD NOT reuse the same player instance concurrently if it has state.
        
        # CHECK: Does Player have state?
        # AlphaZeroPlayer has `self.tree_cache`.
        # So we cannot reuse the same instance.
        
        # We need a way to clone or create new players.
        # For this implementation, let's assume we can clone or the user provided a factory?
        # The current signature takes `players: Dict[str, Player]`.
        # If we pass instances, we have a problem.
        
        # Workaround: If the player has a clone method, use it. Or if it's a factory.
        # Or, we can just disable the tree cache or accept the race condition (bad).
        # Better: Assume the provided players are "templates" and we should try to copy them if possible,
        # or rely on them being thread-safe/async-safe.
        # But `tree_cache` is definitely not async-safe if shared.
        
        # Let's modify the requirement: players should be factories or we clone them.
        # For now, let's try to shallow copy?
        # `copy.copy(player)` might work if `tree_cache` is re-initialized.
        
        # Let's check AlphaZeroPlayer again. `tree_cache` is initialized in `__init__`.
        # If we copy, we get the same cache object.
        # We need a fresh cache.
        
        # Let's assume for now we can just instantiate a new player if we knew how,
        # but we don't know the class or args.
        
        # Hack: Manually reset tree cache on a copy?
        import copy
        
        p1_copy = copy.copy(player1)
        if hasattr(p1_copy, 'tree_cache'):
             # Re-initialize cache
             # We need to import IncrementalTreeCache or just set to a new one if we can access the class.
             # Or just set it to a new instance if we can.
             # Since we are inside `rgi` we can import it?
             # But `Tournament` is in `rgizero`.
             # Let's try to clear it.
             # p1_copy.tree_cache = IncrementalTreeCache()
             # We can't easily import IncrementalTreeCache here without circular imports maybe?
             # Actually `alphazero.py` imports `base.py`. `tournament.py` imports `alphazero.py`.
             # So we can import `IncrementalTreeCache` from `alphazero`.
             pass

        p2_copy = copy.copy(player2)
        
        # We need to handle the cache clearing properly.
        # Let's add a helper method to clear/reset player state if needed.
        # For now, let's just try to run it. If it fails, we fix it.
        # But wait, `play_game_async` runs a loop.
        
        # To make this robust, let's import IncrementalTreeCache and reset it.
        from rgi.rgizero.players.alphazero import IncrementalTreeCache
        
        if hasattr(p1_copy, 'tree_cache'):
            p1_copy.tree_cache = IncrementalTreeCache()
        if hasattr(p2_copy, 'tree_cache'):
            p2_copy.tree_cache = IncrementalTreeCache()

        result = await play_game_async(self.game, [p1_copy, p2_copy])
        
        # result['winner'] is player ID (1 or 2) or None
        winner = result['winner']
        
        score_p1 = 0.5
        if winner == 1:
            score_p1 = 1.0
        elif winner == 2:
            score_p1 = 0.0
            
        self.update_elo(p1_id, p2_id, score_p1)
        return result

    def print_standings(self):
        print("\nTournament Standings:")
        print(f"{'Rank':<5} {'Player':<20} {'ELO':<10} {'Games':<8} {'W-L-D':<15}")
        print("-" * 65)
        
        sorted_players = sorted(self.stats.items(), key=lambda x: x[1].elo, reverse=True)
        
        for rank, (pid, stats) in enumerate(sorted_players, 1):
            wld = f"{stats.wins}-{stats.losses}-{stats.draws}"
            print(f"{rank:<5} {pid:<20} {stats.elo:<10.1f} {stats.games_played:<8} {wld:<15}")
