import json

from arena.artifacts import RunArtifacts
from arena.elo import EloSystem


def make_round(n, success):
    return {
        "run_id": "r1", "round_number": n, "attacker_id": "atk", "defender_id": "dfn",
        "seed_category": "weapons", "judge_attack_success": success,
        "judge_harm_severity": 4, "judge_mode": "heuristic",
    }


def test_streams_rounds_live(tmp_path):
    art = RunArtifacts("r1", "tournament", base_dir=str(tmp_path))
    art.log_round(make_round(1, 1))
    # Flushed immediately — readable mid-run, before finalize.
    lines = (art.dir / "rounds.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["round_number"] == 1
    art.close()


def test_meta_written_at_start(tmp_path):
    art = RunArtifacts("r1", "tournament", run_config={"k": "v"}, base_dir=str(tmp_path))
    meta = json.loads((art.dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["run_id"] == "r1"
    assert meta["config"] == {"k": "v"}
    art.close()


def test_finalize_writes_summary_and_leaderboard(tmp_path):
    art = RunArtifacts("r1", "tournament", base_dir=str(tmp_path))
    for i in range(4):
        art.log_round(make_round(i + 1, success=1 if i < 3 else 0))

    elo = EloSystem()
    elo.register_attacker("atk", "Attacker")
    elo.register_defender("dfn", "Defender")
    elo.update("atk", "dfn", attacker_won=True)

    out = art.finalize(logger=None, elo=elo)  # no logger → no charts, still valid
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))

    assert summary["total_rounds"] == 4
    assert summary["attack_success_rate"] == 0.75
    assert summary["severity_distribution"] == {"4": 4}
    assert summary["judge_modes"] == {"heuristic": 4}
    assert "leaderboard" in summary
    assert (out / "leaderboard.md").exists()


def test_finalize_without_plotting_is_graceful(tmp_path, monkeypatch):
    # Simulate plotly/pandas missing: chart import fails, data still written.
    art = RunArtifacts("r1", "tournament", base_dir=str(tmp_path))
    art.log_round(make_round(1, 1))

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "dashboard" or name.startswith("dashboard."):
            raise ImportError("no plotly")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    class _StubLogger:
        def get_rounds(self, **k): return []
        def get_elo_trajectories(self, rid): return []
        def get_attack_success_matrix(self, rid): return {}
        def get_category_breakdown(self, rid): return []

    out = art.finalize(logger=_StubLogger(), elo=None)
    assert (out / "summary.json").exists()
    assert (out / "CHARTS_SKIPPED.txt").exists()
