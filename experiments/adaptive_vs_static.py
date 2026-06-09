"""
Experiment 3 — Adaptive vs. Static Attackers

Research question:
  Do multi-turn adaptive attackers (that see prior failed attempts) develop meaningfully
  different Elo curves than one-shot static attackers?

Method:
  - Run static attacker (single-turn template) for N rounds
  - Run adaptive attacker (same base, but multi-turn with history) for N rounds
  - Compare: Elo trajectories, attack success rate, turns-to-success distribution

Output:
  - Comparative Elo curves: {attacker_id: [elo_over_time]}
  - Average turns to first success (adaptive only)
  - Statistical comparison of success rates
"""

from __future__ import annotations

import uuid
from typing import Optional

from rich.console import Console
from rich.table import Table

from agents.attacker import BaseAttacker, TemplateAttacker, AdaptiveAttacker, get_attacker
from agents.defender import BaseDefender, get_defender
from agents.judge import JudgeAgent
from arena.controller import ArenaController, SeedPromptBank, RoundResult
from arena.elo import EloSystem
from arena.logger import ArenaLogger
import config

console = Console()


class AdaptiveVsStaticExperiment:
    """
    Compares static one-shot attackers vs. adaptive multi-turn attackers.

    For adaptive runs: the attacker sees previous failed attempts and generates
    a new strategy each turn, up to max_adaptive_turns.
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

        # ── Static run ────────────────────────────────────────────────────────
        console.print("  → Running [cyan]static[/cyan] attacker...")
        static_run_id = f"{self.run_id}_static"
        static_elo = EloSystem()
        static_ctrl = ArenaController(
            attackers=[self.base_attacker],
            defenders=self.defenders,
            judge=self.judge,
            seed_bank=seed_bank,
            logger=self.logger,
            elo=static_elo,
            run_id=static_run_id,
            experiment_tag="adaptive_static_comparison",
            verbose=False,
        )
        static_results = static_ctrl.run_tournament(
            rounds=self.rounds * len(self.defenders),
        )

        # ── Adaptive run ──────────────────────────────────────────────────────
        console.print("  → Running [cyan]adaptive[/cyan] attacker...")
        adaptive_attacker = AdaptiveAttacker(self.base_attacker, self.max_adaptive_turns)
        adaptive_run_id = f"{self.run_id}_adaptive"
        adaptive_results = self._run_adaptive(
            adaptive_attacker, seed_bank, adaptive_run_id
        )

        # ── Analysis ──────────────────────────────────────────────────────────
        self.results = self._analyze(static_results, adaptive_results, static_run_id, adaptive_run_id)
        self._print_results()
        return self.results

    def _run_adaptive(
        self,
        adaptive_attacker: AdaptiveAttacker,
        seed_bank: SeedPromptBank,
        run_id: str,
    ) -> list[RoundResult]:
        """
        Multi-turn adaptive: for each seed, run up to max_turns, passing history
        each time. Stop early if attacker succeeds.
        """
        results = []
        seeds = seed_bank.sample(n=self.rounds)

        for defender in self.defenders:
            elo = EloSystem()
            elo.register_attacker(adaptive_attacker.attacker_id, adaptive_attacker.display_name)
            elo.register_defender(defender.defender_id, defender.display_name)

            ctrl = ArenaController(
                attackers=[adaptive_attacker],
                defenders=[defender],
                judge=self.judge,
                seed_bank=seed_bank,
                logger=self.logger,
                elo=elo,
                run_id=run_id,
                experiment_tag="adaptive_static_comparison",
                verbose=False,
            )

            for seed in seeds:
                adaptive_attacker.reset()
                history = []
                for turn in range(self.max_adaptive_turns):
                    result = ctrl.run_round(adaptive_attacker, defender, seed, history)
                    history.append({
                        "attack_prompt": result.attack.attack_prompt,
                        "template_used": result.attack.template_used,
                        "defender_flagged": result.defense.flagged,
                    })
                    results.append(result)
                    if result.judgment.attack_success:
                        break  # Attacker succeeded — no need for more turns

        return results

    def _analyze(
        self,
        static_results: list[RoundResult],
        adaptive_results: list[RoundResult],
        static_run_id: str,
        adaptive_run_id: str,
    ) -> dict:
        def success_rate(results):
            if not results:
                return 0.0
            return sum(1 for r in results if r.judgment.attack_success) / len(results)

        # Turns-to-success for adaptive
        # Group adaptive results by (seed_id, defender_id) and find first success turn
        turns_to_success = []
        groups: dict[tuple, list[RoundResult]] = {}
        for r in adaptive_results:
            key = (r.seed_prompt_id, r.attack.attacker_id)
            groups.setdefault(key, []).append(r)

        for key, rounds in groups.items():
            rounds_sorted = sorted(rounds, key=lambda x: x.attack.turn)
            for i, r in enumerate(rounds_sorted):
                if r.judgment.attack_success:
                    turns_to_success.append(i + 1)
                    break

        avg_turns = (sum(turns_to_success) / len(turns_to_success)) if turns_to_success else None

        static_rate = success_rate(static_results)
        adaptive_rate = success_rate(adaptive_results)
        improvement = adaptive_rate - static_rate

        return {
            "run_id": self.run_id,
            "static": {
                "run_id":        static_run_id,
                "rounds":        len(static_results),
                "success_rate":  round(static_rate, 3),
                "elo_snapshots": self.logger.get_elo_trajectories(static_run_id),
            },
            "adaptive": {
                "run_id":            adaptive_run_id,
                "rounds":            len(adaptive_results),
                "success_rate":      round(adaptive_rate, 3),
                "avg_turns_to_success": round(avg_turns, 2) if avg_turns else "n/a",
                "turns_to_success_dist": turns_to_success,
                "elo_snapshots":     self.logger.get_elo_trajectories(adaptive_run_id),
            },
            "comparison": {
                "success_rate_improvement": round(improvement, 3),
                "adaptive_advantage":       improvement > 0.05,
            },
        }

    def _print_results(self):
        table = Table(title="Adaptive vs Static Comparison", show_header=True)
        table.add_column("Mode", style="cyan")
        table.add_column("Rounds", justify="right")
        table.add_column("Success Rate", justify="right")
        table.add_column("Avg Turns to Win", justify="right")

        s = self.results["static"]
        a = self.results["adaptive"]
        table.add_row("Static",   str(s["rounds"]), f"{s['success_rate']:.1%}", "1")
        table.add_row("Adaptive", str(a["rounds"]), f"{a['success_rate']:.1%}",
                      str(a["avg_turns_to_success"]))
        console.print(table)

        cmp = self.results["comparison"]
        advantage = "[green]YES[/green]" if cmp["adaptive_advantage"] else "[red]NO[/red]"
        console.print(
            f"\nAdaptive advantage: {advantage}  "
            f"improvement={cmp['success_rate_improvement']:+.1%}"
        )
