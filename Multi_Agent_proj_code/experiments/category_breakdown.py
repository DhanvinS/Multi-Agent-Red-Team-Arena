"""
Experiment 2 — Category Breakdown

Research question:
  Which harm categories have the highest attack success rate against each defender?
  This reveals defender blind spots (categories they systematically miss).

Output:
  - Per-category attack success rate per defender
  - Blind spot map: {defender_id: [categories they most often miss]}
  - Severity-weighted success rate
"""

from __future__ import annotations

import uuid
from typing import Optional

from rich.console import Console
from rich.table import Table

from agents.attacker import BaseAttacker, get_attacker
from agents.defender import BaseDefender, get_defender
from agents.judge import JudgeAgent
from arena.controller import ArenaController, SeedPromptBank
from arena.elo import EloSystem
from arena.logger import ArenaLogger
import config

console = Console()


class CategoryBreakdownExperiment:
    """
    Runs attacks across all harm categories and measures per-category defender accuracy.
    Identifies which categories each defender is weakest against.
    """

    def __init__(
        self,
        attackers: Optional[list[BaseAttacker]] = None,
        defenders: Optional[list[BaseDefender]] = None,
        rounds_per_category: int = 10,
        run_id: Optional[str] = None,
        logger: Optional[ArenaLogger] = None,
        judge: Optional[JudgeAgent] = None,
    ):
        self.attackers = attackers or [
            get_attacker("template_random"),
            get_attacker("template_hypothetical"),
        ]
        self.defenders = defenders or [
            get_defender("keyword_baseline"),
            get_defender("openai_moderation"),
        ]
        self.rounds_per_category = rounds_per_category
        self.run_id = run_id or f"catbreak_{str(uuid.uuid4())[:6]}"
        self.logger = logger or ArenaLogger()
        self.judge = judge or JudgeAgent()
        self.results: dict = {}

    def run(self) -> dict:
        console.print(f"\n[bold magenta]Experiment: Category Breakdown[/bold magenta]  run_id={self.run_id}")

        seed_bank = SeedPromptBank()
        all_rounds = []

        for category in seed_bank.categories:
            console.print(f"  → category: [cyan]{category}[/cyan]")
            elo = EloSystem()
            controller = ArenaController(
                attackers=self.attackers,
                defenders=self.defenders,
                judge=self.judge,
                seed_bank=seed_bank,
                logger=self.logger,
                elo=elo,
                run_id=self.run_id,
                experiment_tag=f"catbreak_{category}",
                verbose=False,
            )
            total = self.rounds_per_category * len(self.attackers) * len(self.defenders)
            rounds = controller.run_tournament(
                rounds=total,
                category_filter=category,
            )
            all_rounds.extend(rounds)

        breakdown = self.logger.get_category_breakdown(self.run_id)
        self.results = self._analyze(breakdown)
        self._print_results()
        return self.results

    def _analyze(self, breakdown: list[dict]) -> dict:
        # Group by defender and find blind spots
        blind_spots: dict[str, list[dict]] = {}
        for row in breakdown:
            dfn = row["defender_id"]
            blind_spots.setdefault(dfn, []).append({
                "category":           row["seed_category"],
                "attack_success_rate": round(row["attack_success_rate"] or 0, 3),
                "defender_accuracy":  round(row["defender_accuracy"] or 0, 3),
                "avg_severity":       round(row["avg_severity"] or 0, 2),
                "total":              row["total"],
            })

        # Sort each defender's categories by attack_success_rate desc
        for dfn in blind_spots:
            blind_spots[dfn].sort(key=lambda x: x["attack_success_rate"], reverse=True)

        return {
            "run_id":      self.run_id,
            "breakdown":   breakdown,
            "blind_spots": blind_spots,
        }

    def _print_results(self):
        table = Table(title="Category × Defender Success Rates", show_header=True)
        table.add_column("Category", style="cyan")
        table.add_column("Defender")
        table.add_column("Attack Success", justify="right")
        table.add_column("Defender Accuracy", justify="right")
        table.add_column("Avg Severity", justify="right")

        for row in self.results.get("breakdown", []):
            table.add_row(
                row.get("seed_category", ""),
                row.get("defender_id", ""),
                f"{(row.get('attack_success_rate') or 0):.1%}",
                f"{(row.get('defender_accuracy') or 0):.1%}",
                f"{(row.get('avg_severity') or 0):.1f}",
            )
        console.print(table)

        # Print blind spot summary
        for dfn, categories in self.results.get("blind_spots", {}).items():
            worst = categories[:3]
            console.print(
                f"\n[bold]{dfn}[/bold] blind spots: "
                + ", ".join(f"[red]{c['category']}[/red] ({c['attack_success_rate']:.0%})" for c in worst)
            )
