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
import json
import random

from rich.console import Console

from agents.attacker import get_attacker
from agents.defender import get_defender, DEFENDER_REGISTRY
from agents.judge import JudgeAgent
from arena.artifacts import RunArtifacts
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
                        help="Use heuristic judge (no API cost)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible runs")
    parser.add_argument("--no-artifacts", action="store_true",
                        help="Disable per-experiment artifact folder (logs + HTML charts)")
    return parser.parse_args()


def _write_artifacts(logger, exp_type: str, run_id: str, run_ids: list, results: dict) -> None:
    """Persist a per-experiment artifact bundle (rounds, summary, results.json,
    charts) built from the SQLite log after the experiment finishes.

    Experiments stream live to SQLite rather than to the artifact folder, so we
    replay the logged rounds through the writer to populate rounds.jsonl and the
    summary tallies before finalizing."""
    art = RunArtifacts(run_id=run_id, experiment_type=exp_type)
    for rid in run_ids:
        for row in sorted(logger.get_rounds(run_id=rid, limit=100000),
                          key=lambda r: (r.get("round_number", 0), r.get("timestamp", 0))):
            art.log_round(row)
    (art.dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    art.finalize(logger=logger, elo=None, run_ids=run_ids, results=results)
    console.print(f"[dim]Artifacts written to[/dim] [cyan]{art.dir}[/cyan]")


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    judge = JudgeAgent(use_llm=False) if args.no_judge else JudgeAgent()
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
        if not args.no_artifacts:
            _write_artifacts(logger, "transferability", exp.run_id,
                             [exp.run_id], results["transferability"])

    if args.experiment in ("category", "all"):
        console.rule("[bold magenta]Experiment 2: Category Breakdown[/bold magenta]")
        exp = CategoryBreakdownExperiment(
            defenders=defenders,
            rounds_per_category=args.rounds,
            logger=logger,
            judge=judge,
        )
        results["category_breakdown"] = exp.run()
        if not args.no_artifacts:
            _write_artifacts(logger, "category_breakdown", exp.run_id,
                             [exp.run_id], results["category_breakdown"])

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
        if not args.no_artifacts:
            r = results["adaptive_vs_static"]
            _write_artifacts(logger, "adaptive_vs_static", exp.run_id,
                             [r["static"]["run_id"], r["adaptive"]["run_id"]], r)

    console.print("\n[bold green]All experiments complete.[/bold green]")
    console.print(f"Results saved to: [cyan]{config.DB_PATH}[/cyan]")
    console.print("Launch dashboard: [cyan]streamlit run dashboard/app.py[/cyan]")


if __name__ == "__main__":
    main()
