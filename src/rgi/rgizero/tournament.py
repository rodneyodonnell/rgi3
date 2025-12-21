import asyncio
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable

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
        player_factories: Dict[str, Callable[[], Player]],
        initial_elo: float = 1200.0,
        k_factor: float = 32.0,
    ):
        self.game = game
        self.player_factories = player_factories
        self.k_factor = k_factor
        self.stats: Dict[str, PlayerStats] = {
            pid: PlayerStats(elo=initial_elo, history=[initial_elo]) for pid in player_factories
        }
        self.match_history: List[Tuple[str, str, float]] = []  # (p1, p2, result for p1)
        self.pending_matches: List[Tuple[str, str]] = []  # (p1, p2) currently running

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
        player_ids = list(self.player_factories.keys())
        if len(player_ids) < 2:
            return []

        # Count games between pairs
        pair_counts = {}
        for p1, p2, _ in self.match_history:
            pair = tuple(sorted((p1, p2)))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        for p1, p2 in self.pending_matches:
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
            best_score = float("inf")

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
        """Run the tournament with constant concurrency."""

        games_to_start = num_games
        active_tasks = set()
        pbar = tqdm(total=num_games, desc="Tournament Progress")

        def start_games():
            nonlocal games_to_start
            needed = concurrent_games - len(active_tasks)
            if needed <= 0 or games_to_start <= 0:
                return

            # Cap needed by games_to_start
            count = min(needed, games_to_start)
            pairings = self.matchmake(count)

            for p1_id, p2_id in pairings:
                # Add to pending
                self.pending_matches.append((p1_id, p2_id))

                # Create wrapper task
                async def game_wrapper(p1, p2):
                    try:
                        await self._play_single_game(p1, p2)
                    finally:
                        # Remove from pending
                        try:
                            self.pending_matches.remove((p1, p2))
                        except ValueError:
                            pass

                task = asyncio.create_task(game_wrapper(p1_id, p2_id))
                active_tasks.add(task)
                games_to_start -= 1

        # Initial start
        start_games()

        while active_tasks:
            done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                await task  # Propagate exceptions if any
                pbar.update(1)

            active_tasks = pending
            start_games()

        pbar.close()

    async def _play_single_game(self, p1_id: str, p2_id: str):
        # Create fresh player instances using the factories
        player1 = self.player_factories[p1_id]()
        player2 = self.player_factories[p2_id]()

        result = await play_game_async(self.game, [player1, player2])

        # result['winner'] is player ID (1 or 2) or None
        winner = result["winner"]

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
