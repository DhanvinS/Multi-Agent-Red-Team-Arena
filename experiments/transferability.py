"""
Experiment 1 — Attacker Transferability

Research question:
  Does an attack that beats Defender A also beat Defender B?
  This reveals whether attack strategies are defender-specific or transfer broadly.

Output:
  - Cross-attack success matrix: {attacker_id: {defender_id: success_rate}}
  - Transferability score per attacker (how consistent their success is across defenders)
  - Heatmap-ready data for the dashboard
"""

from __future__ import annotations

import uuid
from typing import Optional

from rich.console import Console
from rich.table import Table

from agents.attacker import BaseAttacker, get_attacker, ATTACKER_REGISTRY
from agents.defender import BaseDefender, get_defender, DEFENDER_REGISTRY
from agents.judge import JudgeAgent
from arena.controller import ArenaController, SeedPromptBank
from arena.elo import EloSystem
from arena.logger import ArenaLogger
import config

console = Console()


class TransferabilityExperiment:
    """
    Runs every attacker against every defender and measures cross-transfer rates.

    Key metric:
      transferability_score = std_dev of attack_success_rate across defenders
      Low std = attack works uniformly (high transfer)
      High std = attack is defender-specific (low transfer)
    """

    def __init__(
        self,
        attackers: Optional[list[BaseAttacker]] = None,
        defenders: Optional[list[BaseDefender]] = None,
        rounds_per_pair: int = 20,
        category_filter: Optional[str] = None,
        run_id: Optional[str] = None,
        logger: Optional[ArenaLogger] = None,
        judge: Optional[JudgeAgent] = None,
    ):
        # Default: use template attackers (no API needed) and all defenders
        self.attackers = attackers or [
            get_attacker("template_random"),
            get_attacker("template_dan"),
            get_attacker("template_roleplay"),
            get_attacker("template_hypothetical"),
            get_attacker("template_system"),
        ]
        self.defenders = defenders or [
            get_defender("keyword_baseline"),
            get_defender("openai_moderation"),
        ]
        self.rounds_per_pair = rounds_per_pair
        self.category_filter = category_filter
        self.run_id = run_id or f"transfer_{str(uuid.uuid4())[:6]}"
        self.logger = logger or ArenaLogger()
        self.judge = judge or JudgeAgent()
        self.results: dict = {}

    def run(self) -> dict:
        console.print(f"\n[bold magenta]Experiment: Transferability[/bold magenta]  run_id={self.run_id}")

        seed_bank = SeedPromptBank()
        elo = EloSystem()

        controller = ArenaController(
            attackers=self.attackers,
            defenders=self.defenders,
            judge=self.judge,
            seed_bank=seed_bank,
            logger=self.logger,
            elo=elo,
            run_id=self.run_id,
            experiment_tag="transferability",
        )

        total_rounds = self.rounds_per_pair * len(self.attackers) * len(self.defenders)
        controller.run_tournament(
            rounds=total_rounds,
            category_filter=self.category_filter,
        )

        # Compute transferability metrics
        matrix = self.logger.get_attack_success_matrix(self.run_id)
        self.results = self._compute_transferability(matrix)
        self._print_results()
        return self.results

    def _compute_transferability(self, matrix: dict) -> dict:
        import statistics

        transferability = {}
        for attacker_id, defender_scores in matrix.items():
            rates = list(defender_scores.values())
            if len(rates) < 2:
                transferability[attacker_id] = {
                    "success_by_defender": defender_scores,
                    "mean_success_rate":   rates[0] if rates else 0.0,
                    "transferability_score": 0.0,
                    "interpretation": "insufficient data",
                }
                continue

            mean = statistics.mean(rates)
            stdev = statistics.stdev(rates)
            # Low stdev = high transferability (uniform success/failure across defenders)
            interpretation = (
                "high transfer"  if stdev < 0.1 else
                "medium transfer" if stdev < 0.25 else
                "defender-specific"
            )
            transferability[attacker_id] = {
                "success_by_defender":  defender_scores,
                "mean_success_rate":    round(mean, 3),
                "transferability_score": round(1.0 - stdev, 3),  # higher = more transferable
                "stdev":                round(stdev, 3),
                "interpretation":       interpretation,
            }

        return {
            "run_id":          self.run_id,
            "matrix":          matrix,
            "transferability": transferability,
        }

    def _print_results(self):
        table = Table(title="Transferability Results", show_header=True)
        table.add_column("Attacker", style="cyan")
        table.add_column("Mean Success", justify="right")
        table.add_column("Transfer Score", justify="right")
        table.add_column("Interpretation")

        for atk_id, info in self.results.get("transferability", {}).items():
            table.add_row(
                atk_id,
                f"{info['mean_success_rate']:.1%}",
                f"{info['transferability_score']:.3f}",
                info["interpretation"],
            )
        console.print(table)
