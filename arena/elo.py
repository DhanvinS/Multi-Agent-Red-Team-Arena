"""
Elo rating system for the Red Team Arena.

Semantics:
  - Each attacker and each defender has a separate Elo rating.
  - When an attacker "wins" (attack_success=1), attacker rating goes up, defender goes down.
  - When a defender "wins" (attack blocked), defender rating goes up, attacker goes down.
  - Uses standard chess Elo formula with configurable K-factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import config


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class EloRecord:
    entity_id: str
    entity_type: str       # "attacker" or "defender"
    display_name: str
    rating: float = field(default_factory=lambda: config.ELO_INITIAL_RATING)
    wins: int = 0
    losses: int = 0
    draws: int = 0
    history: list[float] = field(default_factory=list)

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "entity_id":    self.entity_id,
            "entity_type":  self.entity_type,
            "display_name": self.display_name,
            "rating":       round(self.rating, 1),
            "wins":         self.wins,
            "losses":       self.losses,
            "draws":        self.draws,
            "games":        self.games,
            "win_rate":     round(self.win_rate, 3),
        }


# ── Elo math ──────────────────────────────────────────────────────────────────

def _expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for player A against player B."""
    return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))


def _new_ratings(
    rating_a: float,
    rating_b: float,
    a_won: bool,
    k: float = config.ELO_K_FACTOR,
) -> tuple[float, float]:
    """Return updated (rating_a, rating_b)."""
    ea = _expected_score(rating_a, rating_b)
    eb = _expected_score(rating_b, rating_a)
    sa = 1.0 if a_won else 0.0
    sb = 1.0 - sa
    return rating_a + k * (sa - ea), rating_b + k * (sb - eb)


# ── Elo system ────────────────────────────────────────────────────────────────

class EloSystem:
    def __init__(
        self,
        k_factor: float = config.ELO_K_FACTOR,
        initial_rating: float = config.ELO_INITIAL_RATING,
    ):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self._attackers: dict[str, EloRecord] = {}
        self._defenders: dict[str, EloRecord] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register_attacker(self, attacker_id: str, display_name: str) -> EloRecord:
        if attacker_id not in self._attackers:
            self._attackers[attacker_id] = EloRecord(
                attacker_id, "attacker", display_name, rating=self.initial_rating
            )
        return self._attackers[attacker_id]

    def register_defender(self, defender_id: str, display_name: str) -> EloRecord:
        if defender_id not in self._defenders:
            self._defenders[defender_id] = EloRecord(
                defender_id, "defender", display_name, rating=self.initial_rating
            )
        return self._defenders[defender_id]

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        attacker_id: str,
        defender_id: str,
        attacker_won: bool,
    ) -> tuple[float, float]:
        """
        Update both ratings given the outcome.
        Returns (new_attacker_rating, new_defender_rating).
        """
        atk = self._attackers[attacker_id]
        dfn = self._defenders[defender_id]

        new_atk, new_dfn = _new_ratings(atk.rating, dfn.rating, attacker_won, self.k_factor)

        # Record history before update
        atk.history.append(atk.rating)
        dfn.history.append(dfn.rating)

        atk.rating = new_atk
        dfn.rating = new_dfn

        if attacker_won:
            atk.wins += 1
            dfn.losses += 1
        else:
            dfn.wins += 1
            atk.losses += 1

        return new_atk, new_dfn

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_attacker(self, attacker_id: str) -> Optional[EloRecord]:
        return self._attackers.get(attacker_id)

    def get_defender(self, defender_id: str) -> Optional[EloRecord]:
        return self._defenders.get(defender_id)

    def attacker_leaderboard(self) -> list[dict]:
        """Return attackers sorted by rating desc."""
        return sorted(
            [r.to_dict() for r in self._attackers.values()],
            key=lambda x: x["rating"],
            reverse=True,
        )

    def defender_leaderboard(self) -> list[dict]:
        """Return defenders sorted by rating desc."""
        return sorted(
            [r.to_dict() for r in self._defenders.values()],
            key=lambda x: x["rating"],
            reverse=True,
        )

    def all_ratings(self) -> dict:
        return {
            "attackers": self.attacker_leaderboard(),
            "defenders": self.defender_leaderboard(),
        }

    def rating_trajectories(self) -> dict[str, list[float]]:
        """Return {entity_id: [rating_over_time]} for all entities."""
        out = {}
        for eid, rec in self._attackers.items():
            out[eid] = rec.history + [rec.rating]
        for eid, rec in self._defenders.items():
            out[eid] = rec.history + [rec.rating]
        return out

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "k_factor":       self.k_factor,
            "initial_rating": self.initial_rating,
            "attackers":      {k: v.__dict__ for k, v in self._attackers.items()},
            "defenders":      {k: v.__dict__ for k, v in self._defenders.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EloSystem":
        elo = cls(data.get("k_factor", config.ELO_K_FACTOR),
                  data.get("initial_rating", config.ELO_INITIAL_RATING))
        for kid, kv in data.get("attackers", {}).items():
            r = EloRecord(**kv)
            elo._attackers[kid] = r
        for kid, kv in data.get("defenders", {}).items():
            r = EloRecord(**kv)
            elo._defenders[kid] = r
        return elo
