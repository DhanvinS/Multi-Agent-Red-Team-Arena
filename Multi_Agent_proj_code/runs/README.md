# Run artifacts

Every tournament (`scripts/run_arena.py`) and experiment (`scripts/run_experiments.py`)
writes a timestamped folder here, e.g. `20260611_143022_tournament_c0864d77/`.

Each folder contains:

| File | Written | Contents |
|---|---|---|
| `meta.json` | at start | run id, agents, config |
| `rounds.jsonl` | **live, one line per round** | full attack/defense/judge record — tail it while a run is in progress |
| `summary.json` | at end | aggregate stats, severity & judge-mode breakdown, leaderboard |
| `leaderboard.md` | at end | human-readable Elo table |
| `results.json` | at end (experiments only) | the experiment's structured metrics |
| `report.html` | at end | index linking the charts |
| `chart_*.html` | at end | interactive Plotly charts (Elo trajectory, attack-success heatmap, category breakdown, severity, leaderboards) |

Open `report.html` in a browser to view the visualizations.

The run folders themselves are git-ignored (they're regenerated output); only this
README and `.gitkeep` are tracked. Disable artifact writing with `--no-artifacts`.
