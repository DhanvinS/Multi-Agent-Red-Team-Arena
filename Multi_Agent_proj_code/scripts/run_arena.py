"""
Main entry point — run a Red Team Arena tournament.

Usage examples:

  # Quick smoke test (no API keys needed)
  python scripts/run_arena.py --attackers template_random template_dan --defenders keyword_baseline --rounds 20 --no-judge

  # Full run with OpenAI moderation defender + Anthropic judge
  python scripts/run_arena.py --rounds 50

  # Specific category focus
  python scripts/run_arena.py --category jailbreaks --rounds 30

  # With Ollama local model
  python scripts/run_arena.py --attackers ollama_llama3 template_random --rounds 40
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import uuid

from rich.console import Console

from agents.attacker import get_attacker, ATTACKER_REGISTRY
from agents.defender import get_defender, DEFENDER_REGISTRY
from agents.judge import JudgeAgent
from arena.controller import ArenaController, SeedPromptBank
from arena.elo import EloSystem
from arena.logger import ArenaLogger
import config

console = Console()

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_ATTACKERS = ["template_random", "template_dan", "template_roleplay", "template_hypothetical"]
DEFAULT_DEFENDERS = ["keyword_baseline", "openai_moderation"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Red Team Arena",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--attackers", nargs="+", default=DEFAULT_ATTACKERS,
                        help=f"Attacker IDs. Available: {list(ATTACKER_REGISTRY.keys())}")
    parser.add_argument("--defenders", nargs="+", default=DEFAULT_DEFENDERS,
                        help=f"Defender IDs. Available: {list(DEFENDER_REGISTRY.keys())}")
    parser.add_argument("--rounds", type=int, default=config.DEFAULT_ROUNDS,
                        help="Total rounds to run")
    parser.add_argument("--category", type=str, default=None,
                        help="Restrict to a specific harm category")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Custom run ID (default: auto-generated)")
    parser.add_argument("--judge-provider", type=str, default=config.JUDGE_PROVIDER,
                        choices=["anthropic", "openai"],
                        help="LLM provider for the judge")
    parser.add_argument("--judge-model", type=str, default=config.JUDGE_MODEL,
                        help="Model ID for the judge")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM judge (use heuristic fallback) — free, no API needed")
    parser.add_argument("--list-agents", action="store_true",
                        help="Print available attackers and defenders then exit")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_agents:
        console.print("\n[bold cyan]Available Attackers:[/bold cyan]")
        for k in ATTACKER_REGISTRY:
            console.print(f"  {k}")
        console.print("\n[bold green]Available Defenders:[/bold green]")
        for k in DEFENDER_REGISTRY:
            console.print(f"  {k}")
        return

    # Resolve agents
    try:
        attackers = [get_attacker(a) for a in args.attackers]
        defenders = [get_defender(d) for d in args.defenders]
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run with --list-agents to see available options.")
        sys.exit(1)

    # Judge setup
    if args.no_judge:
        console.print("[yellow]Judge disabled — using heuristic fallback[/yellow]")
        judge = JudgeAgent(provider="openai", model="heuristic")
        # Monkeypatch to force fallback
        from agents.judge import JudgeAgent as JA
        original_score = JA.score
        def heuristic_score(self, attack_prompt, defender_verdict, seed_category):
            return self._heuristic_verdict(attack_prompt, defender_verdict, seed_category)
        JA.score = heuristic_score
    else:
        judge = JudgeAgent(provider=args.judge_provider, model=args.judge_model)

    run_id = args.run_id or str(uuid.uuid4())[:8]

    console.print(f"\n[bold]Configuration[/bold]")
    console.print(f"  run_id:    [yellow]{run_id}[/yellow]")
    console.print(f"  attackers: {[a.attacker_id for a in attackers]}")
    console.print(f"  defenders: {[d.defender_id for d in defenders]}")
    console.print(f"  rounds:    {args.rounds}")
    console.print(f"  category:  {args.category or 'all'}")
    console.print(f"  judge:     {'heuristic' if args.no_judge else f'{args.judge_provider}/{args.judge_model}'}\n")

    controller = ArenaController(
        attackers=attackers,
        defenders=defenders,
        judge=judge,
        seed_bank=SeedPromptBank(),
        logger=ArenaLogger(),
        elo=EloSystem(),
        run_id=run_id,
    )

    results = controller.run_tournament(
        rounds=args.rounds,
        category_filter=args.category,
    )

    console.print(f"\n[bold green]Done![/bold green] Results saved to: [cyan]{config.DB_PATH}[/cyan]")
    console.print("Launch dashboard: [cyan]streamlit run dashboard/app.py[/cyan]")


if __name__ == "__main__":
    main()
