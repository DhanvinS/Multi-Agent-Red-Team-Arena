"""
Run all three research experiments.

Usage:
  # Run all experiments (no API keys: uses keyword_baseline defender + heuristic judge)
  python scripts/run_experiments.py --no-judge --defenders keyword_baseline

  # Full research run
  python scripts/run_experiments.py --rounds 20

  # Single experiment
  python scripts/run_experiments.py --experiment transferability --rounds 10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from rich.console import Console

from agents.attacker import get_attacker
from agents.defender import get_defender, DEFENDER_REGISTRY
from agents.judge import JudgeAgent
from arena.logger import ArenaLogger
from experiments.transferability import TransferabilityExperiment
from experiments.category_breakdown import CategoryBreakdownExperiment
from experiments.adaptive_vs_static import AdaptiveVsStaticExperiment
import config

console = Console()

DEFAULT_DEFENDERS = ["keyword_baseline", "openai_moderation"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Red Team Arena research experiments")
    parser.add_argument("--experiment", choices=["transferability", "category", "adaptive", "all"],
                        default="all", help="Which experiment to run")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Rounds per pair (keep low for quick runs)")
    parser.add_argument("--defenders", nargs="+", default=DEFAULT_DEFENDERS)
    parser.add_argument("--no-judge", action="store_true",
                        help="Use heuristic fallback (no API cost)")
    return parser.parse_args()


def get_judge(no_judge: bool) -> JudgeAgent:
    if no_judge:
        judge = JudgeAgent(provider="openai", model="heuristic")
        from agents.judge import JudgeAgent as JA
        def heuristic_score(self, attack_prompt, defender_verdict, seed_category):
            return self._heuristic_verdict(attack_prompt, defender_verdict, seed_category)
        JA.score = heuristic_score
        return judge
    return JudgeAgent()


def main():
    args = parse_args()
    judge = get_judge(args.no_judge)
    logger = ArenaLogger()

    try:
        defenders = [get_defender(d) for d in args.defenders]
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    results = {}

    if args.experiment in ("transferability", "all"):
        console.rule("[bold magenta]Experiment 1: Transferability[/bold magenta]")
        exp = TransferabilityExperiment(
            defenders=defenders,
            rounds_per_pair=args.rounds,
            logger=logger,
            judge=judge,
        )
        results["transferability"] = exp.run()

    if args.experiment in ("category", "all"):
        console.rule("[bold magenta]Experiment 2: Category Breakdown[/bold magenta]")
        exp = CategoryBreakdownExperiment(
            defenders=defenders,
            rounds_per_category=args.rounds,
            logger=logger,
            judge=judge,
        )
        results["category_breakdown"] = exp.run()

    if args.experiment in ("adaptive", "all"):
        console.rule("[bold magenta]Experiment 3: Adaptive vs Static[/bold magenta]")
        exp = AdaptiveVsStaticExperiment(
            base_attacker=get_attacker("template_escalating"),
            defenders=defenders,
            rounds=args.rounds,
            logger=logger,
            judge=judge,
        )
        results["adaptive_vs_static"] = exp.run()

    console.print("\n[bold green]All experiments complete.[/bold green]")
    console.print(f"Results saved to: [cyan]{config.DB_PATH}[/cyan]")
    console.print("Launch dashboard: [cyan]streamlit run dashboard/app.py[/cyan]")


if __name__ == "__main__":
    main()
