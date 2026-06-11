import sqlite3
import time

import pytest

from arena.logger import ArenaLogger


@pytest.fixture
def logger(tmp_path):
    return ArenaLogger(db_path=str(tmp_path / "test.db"))


def make_round(**overrides) -> dict:
    base = {
        "run_id": "testrun",
        "timestamp": time.time(),
        "round_number": 1,
        "attacker_id": "atk",
        "defender_id": "dfn",
        "seed_prompt_id": "jb_001",
        "seed_category": "jailbreaks",
        "seed_severity": 3,
        "attack_prompt": "ignore previous instructions",
        "template_used": "dan",
        "defender_flagged": 0,
        "defender_confidence": 0.1,
        "judge_policy_violation": 1,
        "judge_harm_severity": 3,
        "judge_attack_success": 1,
        "judge_defender_correct": 0,
        "judge_harm_category": "jailbreaks",
        "judge_reasoning": "test",
        "judge_confidence": 0.75,
        "judge_mode": "heuristic",
        "attacker_elo_before": 1200.0,
        "attacker_elo_after": 1216.0,
        "defender_elo_before": 1200.0,
        "defender_elo_after": 1184.0,
    }
    base.update(overrides)
    return base


def test_log_round_round_trip(logger):
    logger.log_round(make_round())
    rows = logger.get_rounds(run_id="testrun")
    assert len(rows) == 1
    assert rows[0]["judge_mode"] == "heuristic"
    assert rows[0]["judge_attack_success"] == 1


def test_judge_mode_defaults_to_llm(logger):
    data = make_round()
    del data["judge_mode"]
    logger.log_round(data)
    assert logger.get_rounds(run_id="testrun")[0]["judge_mode"] == "llm"


def test_attack_success_matrix_with_counts(logger):
    logger.log_round(make_round(judge_attack_success=1))
    logger.log_round(make_round(round_number=2, judge_attack_success=0))
    matrix = logger.get_attack_success_matrix("testrun", include_counts=True)
    cell = matrix["atk"]["dfn"]
    assert cell["total"] == 2
    assert cell["success_rate"] == pytest.approx(0.5)
    # Default shape (plain float) is preserved for the dashboard heatmap.
    assert logger.get_attack_success_matrix("testrun")["atk"]["dfn"] == pytest.approx(0.5)


def test_migration_adds_judge_mode_to_old_db(tmp_path):
    db_path = str(tmp_path / "old.db")
    conn = sqlite3.connect(db_path)
    # Pre-judge_mode schema: enough columns for the schema's index creation.
    conn.execute(
        "CREATE TABLE rounds (id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, "
        "attacker_id TEXT, defender_id TEXT, seed_category TEXT, judge_attack_success INTEGER)"
    )
    conn.commit()
    conn.close()

    ArenaLogger(db_path=db_path)

    conn = sqlite3.connect(db_path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(rounds)")}
    conn.close()
    assert "judge_mode" in cols


def test_experiment_lifecycle(logger):
    logger.start_experiment("testrun", "tournament", {"rounds": 5})
    logger.finish_experiment("testrun", 5)
    runs = logger.get_all_run_ids()
    assert runs[0]["run_id"] == "testrun"
    assert runs[0]["total_rounds"] == 5
