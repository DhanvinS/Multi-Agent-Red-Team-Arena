"""
Arena controller — orchestrates the Red Team battle loop.

Each round:
  1. Select a seed prompt
  2. Attacker generates an attack variant
  3. Defender evaluates it
  4. Judge scores the exchange
  5. Elo ratings updated
  6. Everything logged to SQLite
"""

from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

import config
from agents.attacker import BaseAttacker, AttackResult
from agents.defender import BaseDefender, DefenderVerdict
from agents.judge import JudgeAgent, JudgeVerdict
from arena.artifacts import RunArtifacts
from arena.elo import EloSystem
from arena.logger import ArenaLogger


console = Console()


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class RoundResult:
    run_id: str
    round_number: int
    seed_prompt_id: str
    seed_category: str
    seed_severity: int
    attack: AttackResult
    defense: DefenderVerdict
    judgment: JudgeVerdict
    attacker_elo_before: float
    attacker_elo_after: float
    defender_elo_before: float
    defender_elo_after: float

    @property
    def to_log_dict(self) -> dict:
        return {
            "run_id":              self.run_id,
            "timestamp":           time.time(),
            "round_number":        self.round_number,
            "attacker_id":         self.attack.attacker_id,
            "defender_id":         self.defense.defender_id,
            "seed_prompt_id":      self.seed_prompt_id,
            "seed_category":       self.seed_category,
            "seed_severity":       self.seed_severity,
            "attack_prompt":       self.attack.attack_prompt,
            "template_used":       self.attack.template_used,
            "defender_flagged":    int(self.defense.flagged),
            "defender_confidence": self.defense.confidence,
            "defender_latency_ms": self.defense.latency_ms,
            "judge_policy_violation": self.judgment.policy_violation,
            "judge_harm_severity":    self.judgment.harm_severity,
            "judge_attack_success":   self.judgment.attack_success,
            "judge_defender_correct": self.judgment.defender_correct,
            "judge_harm_category":    self.judgment.harm_category,
            "judge_reasoning":        self.judgment.reasoning,
            "judge_confidence":       self.judgment.confidence,
            "judge_mode":             self.judgment.judge_mode,
            "judge_latency_ms":       self.judgment.latency_ms,
            "attacker_elo_before": self.attacker_elo_before,
            "attacker_elo_after":  self.attacker_elo_after,
            "defender_elo_before": self.defender_elo_before,
            "defender_elo_after":  self.defender_elo_after,
            "turn_number":         self.attack.turn,
        }


# ── Seed loader ───────────────────────────────────────────────────────────────

class SeedPromptBank:
    def __init__(self, path: str = config.SEED_PROMPTS_PATH):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.prompts: list[dict] = data["prompts"]
        self._by_category: dict[str, list[dict]] = {}
        for p in self.prompts:
            self._by_category.setdefault(p["category"], []).append(p)

    def sample(
        self,
        category: Optional[str] = None,
        exclude_ids: Optional[set] = None,
        n: int = 1,
    ) -> list[dict]:
        pool = self._by_category.get(category, self.prompts) if category else self.prompts
        if exclude_ids:
            pool = [p for p in pool if p["id"] not in exclude_ids]
        return random.sample(pool, min(n, len(pool)))

    @property
    def categories(self) -> list[str]:
        return list(self._by_category.keys())

    def __len__(self) -> int:
        return len(self.prompts)


# ── Arena controller ──────────────────────────────────────────────────────────

class ArenaController:
    def __init__(
        self,
        attackers: list[BaseAttacker],
        defenders: list[BaseDefender],
        judge: Optional[JudgeAgent] = None,
        seed_bank: Optional[SeedPromptBank] = None,
        logger: Optional[ArenaLogger] = None,
        elo: Optional[EloSystem] = None,
        run_id: Optional[str] = None,
        experiment_tag: str = "",
        verbose: bool = True,
        artifacts: Optional[RunArtifacts] = None,
        write_artifacts: bool = False,
    ):
        self.attackers = {a.attacker_id: a for a in attackers}
        self.defenders = {d.defender_id: d for d in defenders}
        self.judge = judge or JudgeAgent()
        self.seed_bank = seed_bank or SeedPromptBank()
        self.logger = logger or ArenaLogger()
        self.elo = elo or EloSystem()
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.experiment_tag = experiment_tag
        self.verbose = verbose
        self._round_count = 0

        # Per-run artifact folder. An externally supplied writer is shared and
        # NOT finalized here (the caller owns it); an auto-created one is owned
        # and finalized at the end of run_tournament.
        self.artifacts = artifacts
        self._owns_artifacts = False
        if self.artifacts is None and write_artifacts:
            self.artifacts = RunArtifacts(
                run_id=self.run_id,
                experiment_type=experiment_tag or "tournament",
                run_config={
                    "attackers": list(self.attackers),
                    "defenders": list(self.defenders),
                },
            )
            self._owns_artifacts = True

        # Register all participants with Elo
        for a in attackers:
            self.elo.register_attacker(a.attacker_id, a.display_name)
        for d in defenders:
            self.elo.register_defender(d.defender_id, d.display_name)

    # ── Single round ──────────────────────────────────────────────────────────

    def run_round(
        self,
        attacker: BaseAttacker,
        defender: BaseDefender,
        seed: Optional[dict] = None,
        conversation_history: Optional[list] = None,
        update_elo: bool = True,
    ) -> RoundResult:
        """Run one attack/defense/judge exchange.

        update_elo=False lets multi-turn experiments score an episode of
        several turns as a single Elo game instead of one game per turn.
        """
        self._round_count += 1

        if seed is None:
            seed = self.seed_bank.sample()[0]

        # 1. Generate attack
        attack_result = attacker.generate_attack(
            seed["text"], seed["category"], conversation_history
        )

        # 2. Defender evaluates
        defense_verdict = defender.evaluate(attack_result.attack_prompt)

        # 3. Judge scores
        judge_verdict = self.judge.score(
            attack_result.attack_prompt,
            defense_verdict,
            seed["category"],
            seed_severity=seed.get("severity", 3),
        )

        # 4. Elo update
        atk_elo_before = self.elo.get_attacker(attacker.attacker_id).rating
        dfn_elo_before = self.elo.get_defender(defender.defender_id).rating

        attacker_won = bool(judge_verdict.attack_success)
        if update_elo:
            new_atk_elo, new_dfn_elo = self.elo.update(
                attacker.attacker_id, defender.defender_id, attacker_won
            )
        else:
            new_atk_elo, new_dfn_elo = atk_elo_before, dfn_elo_before

        # 5. Build result
        result = RoundResult(
            run_id=self.run_id,
            round_number=self._round_count,
            seed_prompt_id=seed["id"],
            seed_category=seed["category"],
            seed_severity=seed["severity"],
            attack=attack_result,
            defense=defense_verdict,
            judgment=judge_verdict,
            attacker_elo_before=atk_elo_before,
            attacker_elo_after=new_atk_elo,
            defender_elo_before=dfn_elo_before,
            defender_elo_after=new_dfn_elo,
        )

        # 6. Log — to SQLite, and stream to the run-artifact folder if active
        log_dict = result.to_log_dict
        log_dict["experiment_tag"] = self.experiment_tag
        self.logger.log_round(log_dict)
        if self.artifacts is not None:
            self.artifacts.log_round(log_dict)

        return result

    # ── Tournament ────────────────────────────────────────────────────────────

    def run_tournament(
        self,
        rounds: int = config.DEFAULT_ROUNDS,
        category_filter: Optional[str] = None,
        snapshot_every: int = 10,
    ) -> list[RoundResult]:
        """
        Run a round-robin tournament: each attacker faces each defender
        for `rounds // (n_attackers * n_defenders)` rounds.
        """
        attacker_list = list(self.attackers.values())
        defender_list = list(self.defenders.values())
        n_pairs = len(attacker_list) * len(defender_list)
        rounds_per_pair = max(1, rounds // n_pairs)

        self.logger.start_experiment(
            self.run_id,
            "tournament",
            {
                "attackers": [a.attacker_id for a in attacker_list],
                "defenders": [d.defender_id for d in defender_list],
                "rounds_per_pair": rounds_per_pair,
                "category_filter": category_filter,
            },
        )

        results: list[RoundResult] = []
        total_rounds = rounds_per_pair * n_pairs

        if self.verbose:
            console.print(
                f"\n[bold cyan]Red Team Arena[/bold cyan] — "
                f"run_id=[yellow]{self.run_id}[/yellow]  "
                f"pairs={n_pairs}  rounds={total_rounds}"
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            disable=not self.verbose,
        ) as progress:
            task = progress.add_task("Running rounds...", total=total_rounds)

            for attacker in attacker_list:
                for defender in defender_list:
                    used_seeds: set[str] = set()
                    for i in range(rounds_per_pair):
                        # Tournament rounds are independent single-turn games:
                        # adaptive attackers must not carry history across seeds.
                        reset = getattr(attacker, "reset", None)
                        if callable(reset):
                            reset()
                        sampled = self.seed_bank.sample(
                            category=category_filter, exclude_ids=used_seeds
                        )
                        if not sampled:
                            # All seeds exhausted for this pair — reset exclusion and resample
                            used_seeds.clear()
                            sampled = self.seed_bank.sample(category=category_filter)
                        seed = sampled[0]
                        used_seeds.add(seed["id"])
                        result = self.run_round(attacker, defender, seed)
                        results.append(result)

                        # Periodic Elo snapshot
                        if len(results) % snapshot_every == 0:
                            self._snapshot_elo()

                        progress.advance(task)

        self.logger.finish_experiment(self.run_id, len(results))
        self._snapshot_elo()  # final snapshot so the trajectory chart ends at the true rating
        if self.verbose:
            self._print_summary(results)

        # Finalize an auto-created artifact bundle (summary + charts on disk).
        if self.artifacts is not None and self._owns_artifacts:
            out_dir = self.artifacts.finalize(logger=self.logger, elo=self.elo,
                                              run_ids=[self.run_id])
            if self.verbose:
                console.print(f"[dim]Artifacts written to[/dim] [cyan]{out_dir}[/cyan]")

        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _snapshot_elo(self):
        for a_id, rec in self.elo._attackers.items():
            self.logger.log_elo_snapshot(self.run_id, a_id, "attacker", rec.rating, rec.games)
        for d_id, rec in self.elo._defenders.items():
            self.logger.log_elo_snapshot(self.run_id, d_id, "defender", rec.rating, rec.games)

    def _print_summary(self, results: list[RoundResult]):
        total = len(results)
        wins = sum(1 for r in results if r.judgment.attack_success)
        console.print(f"\n[bold]Run complete[/bold]  rounds={total}  "
                      f"attack_success_rate=[yellow]{wins/total:.1%}[/yellow]")

        # Elo leaderboard
        atk_table = Table(title="Attacker Elo", show_header=True)
        atk_table.add_column("Attacker")
        atk_table.add_column("Rating", justify="right")
        atk_table.add_column("W/L", justify="right")
        for row in self.elo.attacker_leaderboard():
            atk_table.add_row(
                row["display_name"],
                f"{row['rating']:.0f}",
                f"{row['wins']}/{row['losses']}",
            )
        console.print(atk_table)

        dfn_table = Table(title="Defender Elo", show_header=True)
        dfn_table.add_column("Defender")
        dfn_table.add_column("Rating", justify="right")
        dfn_table.add_column("W/L", justify="right")
        for row in self.elo.defender_leaderboard():
            dfn_table.add_row(
                row["display_name"],
                f"{row['rating']:.0f}",
                f"{row['wins']}/{row['losses']}",
            )
        console.print(dfn_table)

    def get_leaderboard(self) -> dict:
        return self.elo.all_ratings()

    def get_attack_matrix(self) -> dict:
        return self.logger.get_attack_success_matrix(self.run_id)
