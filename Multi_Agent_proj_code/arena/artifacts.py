"""
Per-run artifact writer.

Every run gets its own timestamped folder under `runs/`:

  runs/20260611_143022_tournament_c0864d77/
    meta.json          — run config + agents, written at start
    rounds.jsonl       — one JSON line per round, appended live as rounds finish
    summary.json       — aggregate stats, written at finalize
    leaderboard.md     — human-readable Elo leaderboard
    report.html        — index linking the charts below
    chart_*.html       — interactive Plotly visualizations

`rounds.jsonl` is flushed after every round, so the folder is a faithful,
real-time record of a run in progress — tail it while a tournament runs.
Charts are written once at the end (they need the full run to be meaningful).

Plotting degrades gracefully: if plotly/pandas are missing, the JSON/markdown
artifacts are still written and chart generation is skipped with a warning.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import config


class RunArtifacts:
    def __init__(
        self,
        run_id: str,
        experiment_type: str = "tournament",
        run_config: Optional[dict] = None,
        base_dir: Optional[str] = None,
    ):
        self.run_id = run_id
        self.experiment_type = experiment_type
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = Path(base_dir or config.RUNS_DIR)
        self.dir = base / f"{ts}_{experiment_type}_{run_id}"
        self.dir.mkdir(parents=True, exist_ok=True)

        self._rounds_path = self.dir / "rounds.jsonl"
        self._fh = open(self._rounds_path, "a", encoding="utf-8")

        # Running tallies so a partial run still has a usable summary.
        self._n = 0
        self._wins = 0
        self._severity = Counter()
        self._judge_modes = Counter()
        self._started = time.time()

        (self.dir / "meta.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "experiment_type": experiment_type,
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "config": run_config or {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # ── Live logging ──────────────────────────────────────────────────────────

    def log_round(self, round_dict: dict) -> None:
        """Append one round to rounds.jsonl and flush immediately."""
        self._fh.write(json.dumps(round_dict, default=str) + "\n")
        self._fh.flush()
        self._n += 1
        self._wins += int(round_dict.get("judge_attack_success", 0))
        self._severity[round_dict.get("judge_harm_severity", 1)] += 1
        self._judge_modes[round_dict.get("judge_mode", "llm")] += 1

    # ── Finalization ──────────────────────────────────────────────────────────

    def finalize(
        self,
        logger=None,
        elo=None,
        run_ids: Optional[list[str]] = None,
        results: Optional[dict] = None,
    ) -> Path:
        """Write summary, leaderboard, and charts. Returns the run folder path."""
        try:
            self._fh.flush()
        except ValueError:
            pass  # already closed

        summary = {
            "run_id": self.run_id,
            "experiment_type": self.experiment_type,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(time.time() - self._started, 2),
            "total_rounds": self._n,
            "attack_success_rate": round(self._wins / self._n, 4) if self._n else 0.0,
            "severity_distribution": {str(k): v for k, v in sorted(self._severity.items())},
            "judge_modes": dict(self._judge_modes),
        }
        if elo is not None:
            summary["leaderboard"] = elo.all_ratings()
        if results is not None:
            summary["experiment_results"] = results

        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._write_leaderboard_md(summary, elo)

        if logger is not None:
            self._write_charts(logger, elo, run_ids or [self.run_id])

        self.close()
        return self.dir

    def close(self) -> None:
        try:
            self._fh.close()
        except (ValueError, OSError):
            pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _write_leaderboard_md(self, summary: dict, elo) -> None:
        lines = [
            f"# Run `{self.run_id}` — {self.experiment_type}",
            "",
            f"- Rounds: **{summary['total_rounds']}**",
            f"- Attack success rate: **{summary['attack_success_rate']:.1%}**",
            f"- Duration: {summary['duration_s']}s",
            f"- Judge modes: {summary['judge_modes']}",
            "",
        ]
        if elo is not None:
            for kind, board in (("Attacker", elo.attacker_leaderboard()),
                                ("Defender", elo.defender_leaderboard())):
                lines += [f"## {kind} Elo", "", "| Agent | Rating | W | L |", "|---|---|---|---|"]
                for r in board:
                    lines.append(
                        f"| {r['display_name']} | {r['rating']:.0f} | {r['wins']} | {r['losses']} |"
                    )
                lines.append("")
        (self.dir / "leaderboard.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_charts(self, logger, elo, run_ids: list[str]) -> None:
        try:
            from dashboard import charts
        except Exception as exc:  # plotly/pandas missing — skip charts, keep data
            (self.dir / "CHARTS_SKIPPED.txt").write_text(
                f"Chart generation skipped: {exc}\n"
                "Install plotly + pandas to enable HTML visualizations.\n",
                encoding="utf-8",
            )
            return

        # Gather data across all run_ids that belong to this artifact bundle.
        rounds: list[dict] = []
        trajectories: list[dict] = []
        for rid in run_ids:
            rounds.extend(logger.get_rounds(run_id=rid, limit=100000))
            trajectories.extend(logger.get_elo_trajectories(rid))
        matrix = logger.get_attack_success_matrix(run_ids[0]) if run_ids else {}
        breakdown = logger.get_category_breakdown(run_ids[0]) if run_ids else []

        written: list[tuple[str, str]] = []

        def _save(name: str, fig, title: str):
            path = self.dir / f"chart_{name}.html"
            fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
            written.append((path.name, title))

        try:
            if trajectories:
                _save("elo_trajectory", charts.elo_trajectory_chart(trajectories), "Elo Over Time")
            if matrix:
                _save("attack_heatmap", charts.attack_success_heatmap(matrix), "Attack Success Heatmap")
            if breakdown:
                _save("category_breakdown", charts.category_bar_chart(breakdown), "Category Breakdown")
            if rounds:
                _save("severity", charts.severity_distribution(rounds), "Harm Severity Distribution")
            if elo is not None:
                _save("attacker_elo", charts.elo_leaderboard_bar(elo.attacker_leaderboard(), "attacker"), "Attacker Elo")
                _save("defender_elo", charts.elo_leaderboard_bar(elo.defender_leaderboard(), "defender"), "Defender Elo")
        except Exception as exc:
            (self.dir / "CHARTS_PARTIAL.txt").write_text(f"Some charts failed: {exc}\n", encoding="utf-8")

        self._write_report_index(written)

    def _write_report_index(self, charts_written: list[tuple[str, str]]) -> None:
        items = "\n".join(
            f'      <li><a href="{fname}">{title}</a></li>' for fname, title in charts_written
        )
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Red Team Arena — {self.run_id}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background:#111; color:#eee; max-width:760px; margin:40px auto; padding:0 16px; }}
    a {{ color:#ff7a59; }} h1 {{ font-weight:600; }}
    code {{ background:#222; padding:2px 6px; border-radius:4px; }}
  </style>
</head>
<body>
  <h1>Red Team Arena — run <code>{self.run_id}</code></h1>
  <p>Experiment: <code>{self.experiment_type}</code></p>
  <p>Artifacts: <code>summary.json</code>, <code>rounds.jsonl</code>, <code>leaderboard.md</code></p>
  <h2>Visualizations</h2>
  <ul>
{items if items else "      <li>(no charts generated)</li>"}
  </ul>
</body>
</html>"""
        (self.dir / "report.html").write_text(html, encoding="utf-8")
