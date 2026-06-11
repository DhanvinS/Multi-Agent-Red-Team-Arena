"""
Experiment 3 — Adaptive vs. Static Attackers

Research question:
  Do multi-turn adaptive attackers (that see prior failed attempts) outperform
  one-shot static attackers?

Method (paired comparison):
  - Sample ONE set of seeds, used by both arms against the same defenders.
  - Static arm: one attempt per (seed, defender).
  - Adaptive arm: up to max_adaptive_turns attempts per (seed, defender),
    passing failure history each turn; stop early on success.
  - Compare PER-SEED success rates (did the attacker succeed at all for this
    seed?) — comparing per-attempt rates would penalize the adaptive arm for
    its own failed intermediate turns.
  - Elo is updated once per episode in the adaptive arm, not once per turn,
    so defender ratings aren't inflated by intermediate failures.

Output:
  - Per-seed success rate for both arms + two-proportion z-test
  - Turns-to-success distribution (adaptive only)
  - Elo snapshots per arm
"""

from __future__ import annotations

import math
import uuid
from typing import Optional

from rich.console import Console
from rich.table import Table

from agents.attacker import BaseAttacker, AdaptiveAttacker, get_attacker
from agents.defender import BaseDefender, get_defender
from agents.judge import JudgeAgent
from arena.controller import ArenaController, SeedPromptBank, RoundResult
from arena.elo import EloSystem
from arena.logger import ArenaLogger
import config

console = Console()


def two_proportion_ztest(successes_a: int, n_a: int, successes_b: int, n_b: int) -> tuple[float, float]:
    """Two-sided two-proportion z-test. Returns (z, p_value)."""
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0
    p_pool = (successes_a + successes_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0, 1.0
    z = (successes_b / n_b - successes_a / n_a) / se
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return z, p_value


class AdaptiveVsStaticExperiment:
    """
    Compares static one-shot attackers vs. adaptive multi-turn attackers on
    the same seeds against the same defenders.
    """

    def __init__(
        self,
        base_attacker: Optional[BaseAttacker] = None,
        defenders: Optional[list[BaseDefender]] = None,
        rounds: int = 30,
        max_adaptive_turns: int = 5,
        run_id: Optional[str] = None,
        logger: Optional[ArenaLogger] = None,
        judge: Optional[JudgeAgent] = None,
    ):
        self.base_attacker = base_attacker or get_attacker("template_random")
        self.defenders = defenders or [
            get_defender("keyword_baseline"),
            get_defender("openai_moderation"),
        ]
        self.rounds = rounds
        self.max_adaptive_turns = max_adaptive_turns
        self.run_id = run_id or f"adaptive_{str(uuid.uuid4())[:6]}"
        self.logger = logger or ArenaLogger()
        self.judge = judge or JudgeAgent()
        self.results: dict = {}

    def run(self) -> dict:
        console.print(f"\n[bold magenta]Experiment: Adaptive vs Static[/bold magenta]  run_id={self.run_id}")

        seed_bank = SeedPromptBank()
        # One seed set, shared by both arms — paired comparison.
        seeds = seed_bank.sample(n=self.rounds)

        console.print("  -> Running [cyan]static[/cyan] attacker...")
        static_run_id = f"{self.run_id}_static"
        static_results = self._run_static(seeds, seed_bank, static_run_id)

        console.print("  -> Running [cyan]adaptive[/cyan] attacker...")
        adaptive_run_id = f"{self.run_id}_adaptive"
        episodes = self._run_adaptive(seeds, seed_bank, adaptive_run_id)

        self.results = self._analyze(static_results, episodes, static_run_id, adaptive_run_id)
        self._print_results()
        return self.results

    def _run_static(
        self,
        seeds: list[dict],
        seed_bank: SeedPromptBank,
        run_id: str,
    ) -> list[RoundResult]:
        """One attempt per (seed, defender) — the same seeds the adaptive arm gets."""
        ctrl = ArenaController(
            attackers=[self.base_attacker],
            defenders=self.defenders,
            judge=self.judge,
            seed_bank=seed_bank,
            logger=self.logger,
            elo=EloSystem(),
            run_id=run_id,
            experiment_tag="adaptive_static_comparison",
            verbose=False,
        )
        self.logger.start_experiment(
            run_id, "adaptive_static_comparison",
            {"arm": "static", "seeds": len(seeds),
             "defenders": [d.defender_id for d in self.defenders]},
        )

        results = []
        for defender in self.defenders:
            for seed in seeds:
                results.append(ctrl.run_round(self.base_attacker, defender, seed))

        self.logger.finish_experiment(run_id, len(results))
        return results

    def _run_adaptive(
        self,
        seeds: list[dict],
        seed_bank: SeedPromptBank,
        run_id: str,
    ) -> list[dict]:
        """
        Multi-turn adaptive: for each (seed, defender) episode, run up to
        max_turns passing failure history each time; stop early on success.
        Elo is updated once per episode with the episode outcome.
        """
        adaptive_attacker = AdaptiveAttacker(self.base_attacker, self.max_adaptive_turns)
        ctrl = ArenaController(
            attackers=[adaptive_attacker],
            defenders=self.defenders,
            judge=self.judge,
            seed_bank=seed_bank,
            logger=self.logger,
            elo=EloSystem(),
            run_id=run_id,
            experiment_tag="adaptive_static_comparison",
            verbose=False,
        )
        self.logger.start_experiment(
            run_id, "adaptive_static_comparison",
            {"arm": "adaptive", "seeds": len(seeds),
             "max_turns": self.max_adaptive_turns,
             "defenders": [d.defender_id for d in self.defenders]},
        )

        episodes: list[dict] = []
        total_turns = 0
        for defender in self.defenders:
            for seed in seeds:
                adaptive_attacker.reset()
                history: list[dict] = []
                turn_results: list[RoundResult] = []
                succeeded = False

                for _ in range(self.max_adaptive_turns):
                    result = ctrl.run_round(
                        adaptive_attacker, defender, seed, history, update_elo=False
                    )
                    history.append({
                        "attack_prompt": result.attack.attack_prompt,
                        "template_used": result.attack.template_used,
                        "defender_flagged": result.defense.flagged,
                    })
                    turn_results.append(result)
                    if result.judgment.attack_success:
                        succeeded = True
                        break

                # One Elo game per episode, scored on the episode outcome.
                ctrl.elo.update(
                    adaptive_attacker.attacker_id, defender.defender_id, succeeded
                )

                total_turns += len(turn_results)
                episodes.append({
                    "seed_id":          seed["id"],
                    "defender_id":      defender.defender_id,
                    "turns":            len(turn_results),
                    "succeeded":        succeeded,
                    "turns_to_success": len(turn_results) if succeeded else None,
                })

        self.logger.finish_experiment(run_id, total_turns)
        return episodes

    def _analyze(
        self,
        static_results: list[RoundResult],
        episodes: list[dict],
        static_run_id: str,
        adaptive_run_id: str,
    ) -> dict:
        # Static arm: one attempt per seed, so per-attempt == per-seed.
        static_n = len(static_results)
        static_successes = sum(1 for r in static_results if r.judgment.attack_success)
        static_rate = static_successes / static_n if static_n else 0.0

        # Adaptive arm: per-seed (episode) success is the fair comparison.
        adaptive_n = len(episodes)
        adaptive_successes = sum(1 for e in episodes if e["succeeded"])
        adaptive_rate = adaptive_successes / adaptive_n if adaptive_n else 0.0

        total_turns = sum(e["turns"] for e in episodes)
        per_attempt_rate = adaptive_successes / total_turns if total_turns else 0.0

        turns_to_success = [e["turns_to_success"] for e in episodes if e["succeeded"]]
        avg_turns = (sum(turns_to_success) / len(turns_to_success)) if turns_to_success else None

        z, p_value = two_proportion_ztest(
            static_successes, static_n, adaptive_successes, adaptive_n
        )
        improvement = adaptive_rate - static_rate

        return {
            "run_id": self.run_id,
            "static": {
                "run_id":        static_run_id,
                "episodes":      static_n,
                "success_rate":  round(static_rate, 3),
                "elo_snapshots": self.logger.get_elo_trajectories(static_run_id),
            },
            "adaptive": {
                "run_id":               adaptive_run_id,
                "episodes":             adaptive_n,
                "total_turns":          total_turns,
                "success_rate":         round(adaptive_rate, 3),
                "per_attempt_success_rate": round(per_attempt_rate, 3),
                "avg_turns_to_success": round(avg_turns, 2) if avg_turns else None,
                "turns_to_success_dist": turns_to_success,
                "elo_snapshots":        self.logger.get_elo_trajectories(adaptive_run_id),
            },
            "comparison": {
                "success_rate_improvement": round(improvement, 3),
                "z_statistic":              round(z, 3),
                "p_value":                  round(p_value, 4),
                "significant_at_05":        p_value < 0.05,
                "adaptive_advantage":       improvement > 0 and p_value < 0.05,
            },
        }

    def _print_results(self):
        table = Table(title="Adaptive vs Static Comparison (per-seed)", show_header=True)
        table.add_column("Mode", style="cyan")
        table.add_column("Episodes", justify="right")
        table.add_column("Success Rate", justify="right")
        table.add_column("Avg Turns to Win", justify="right")

        s = self.results["static"]
        a = self.results["adaptive"]
        avg_turns = a["avg_turns_to_success"]
        table.add_row("Static",   str(s["episodes"]), f"{s['success_rate']:.1%}", "1")
        table.add_row("Adaptive", str(a["episodes"]), f"{a['success_rate']:.1%}",
                      str(avg_turns) if avg_turns else "n/a")
        console.print(table)

        cmp = self.results["comparison"]
        advantage = "[green]YES[/green]" if cmp["adaptive_advantage"] else "[red]NO[/red]"
        console.print(
            f"\nAdaptive advantage: {advantage}  "
            f"improvement={cmp['success_rate_improvement']:+.1%}  "
            f"z={cmp['z_statistic']}  p={cmp['p_value']}"
            + ("" if cmp["significant_at_05"] else "  [dim](not significant at alpha=0.05)[/dim]")
        )
