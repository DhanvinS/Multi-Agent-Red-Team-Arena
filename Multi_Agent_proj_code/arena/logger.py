"""
SQLite-based logger for all arena rounds and Elo state.

Schema:
  rounds        — one row per arena round with full attack/defense/judge data
  elo_snapshots — periodic Elo rating snapshots for trajectory analysis
  experiments   — high-level experiment run metadata
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import config


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS rounds (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT    NOT NULL,
    experiment_tag      TEXT    DEFAULT '',
    timestamp           REAL    NOT NULL,
    round_number        INTEGER NOT NULL,
    attacker_id         TEXT    NOT NULL,
    defender_id         TEXT    NOT NULL,
    seed_prompt_id      TEXT    NOT NULL,
    seed_category       TEXT    NOT NULL,
    seed_severity       INTEGER NOT NULL,
    attack_prompt       TEXT    NOT NULL,
    template_used       TEXT    DEFAULT '',
    defender_flagged    INTEGER NOT NULL,
    defender_confidence REAL    NOT NULL,
    defender_latency_ms REAL    DEFAULT 0,
    judge_policy_violation  INTEGER NOT NULL,
    judge_harm_severity     INTEGER NOT NULL,
    judge_attack_success    INTEGER NOT NULL,
    judge_defender_correct  INTEGER NOT NULL,
    judge_harm_category     TEXT    NOT NULL,
    judge_reasoning         TEXT    NOT NULL,
    judge_confidence        REAL    NOT NULL,
    judge_mode              TEXT    DEFAULT 'llm',
    judge_latency_ms        REAL    DEFAULT 0,
    attacker_elo_before REAL    DEFAULT 0,
    attacker_elo_after  REAL    DEFAULT 0,
    defender_elo_before REAL    DEFAULT 0,
    defender_elo_after  REAL    DEFAULT 0,
    turn_number         INTEGER DEFAULT 1,
    metadata            TEXT    DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS elo_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT    NOT NULL,
    timestamp   REAL    NOT NULL,
    entity_id   TEXT    NOT NULL,
    entity_type TEXT    NOT NULL,
    rating      REAL    NOT NULL,
    games       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL UNIQUE,
    experiment_type TEXT    NOT NULL,
    config_json     TEXT    NOT NULL,
    started_at      REAL    NOT NULL,
    finished_at     REAL    DEFAULT NULL,
    total_rounds    INTEGER DEFAULT 0,
    notes           TEXT    DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_rounds_run       ON rounds(run_id);
CREATE INDEX IF NOT EXISTS idx_rounds_attacker  ON rounds(attacker_id);
CREATE INDEX IF NOT EXISTS idx_rounds_defender  ON rounds(defender_id);
CREATE INDEX IF NOT EXISTS idx_rounds_category  ON rounds(seed_category);
CREATE INDEX IF NOT EXISTS idx_rounds_success   ON rounds(judge_attack_success);
"""


# ── Logger ────────────────────────────────────────────────────────────────────

class ArenaLogger:
    def __init__(self, db_path: str = config.DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            # Migration: judge_mode was added after the original schema;
            # CREATE TABLE IF NOT EXISTS won't add it to existing databases.
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(rounds)")}
            if "judge_mode" not in cols:
                conn.execute("ALTER TABLE rounds ADD COLUMN judge_mode TEXT DEFAULT 'llm'")

    # ── Writes ────────────────────────────────────────────────────────────────

    def log_round(self, round_data: dict) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO rounds (
                    run_id, experiment_tag, timestamp, round_number,
                    attacker_id, defender_id,
                    seed_prompt_id, seed_category, seed_severity,
                    attack_prompt, template_used,
                    defender_flagged, defender_confidence, defender_latency_ms,
                    judge_policy_violation, judge_harm_severity,
                    judge_attack_success, judge_defender_correct,
                    judge_harm_category, judge_reasoning, judge_confidence,
                    judge_mode, judge_latency_ms,
                    attacker_elo_before, attacker_elo_after,
                    defender_elo_before, defender_elo_after,
                    turn_number, metadata
                ) VALUES (
                    :run_id, :experiment_tag, :timestamp, :round_number,
                    :attacker_id, :defender_id,
                    :seed_prompt_id, :seed_category, :seed_severity,
                    :attack_prompt, :template_used,
                    :defender_flagged, :defender_confidence, :defender_latency_ms,
                    :judge_policy_violation, :judge_harm_severity,
                    :judge_attack_success, :judge_defender_correct,
                    :judge_harm_category, :judge_reasoning, :judge_confidence,
                    :judge_mode, :judge_latency_ms,
                    :attacker_elo_before, :attacker_elo_after,
                    :defender_elo_before, :defender_elo_after,
                    :turn_number, :metadata
                )""",
                {
                    "experiment_tag":   round_data.get("experiment_tag", ""),
                    "metadata":         json.dumps(round_data.get("metadata", {})),
                    "defender_latency_ms": round_data.get("defender_latency_ms", 0),
                    "judge_mode":          round_data.get("judge_mode", "llm"),
                    "judge_latency_ms":    round_data.get("judge_latency_ms", 0),
                    "template_used":       round_data.get("template_used", ""),
                    "turn_number":         round_data.get("turn_number", 1),
                    **round_data,
                },
            )
            return cur.lastrowid

    def log_elo_snapshot(self, run_id: str, entity_id: str, entity_type: str,
                         rating: float, games: int):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO elo_snapshots (run_id, timestamp, entity_id, entity_type, rating, games) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, time.time(), entity_id, entity_type, rating, games),
            )

    def start_experiment(self, run_id: str, experiment_type: str, cfg: dict, notes: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO experiments (run_id, experiment_type, config_json, started_at, notes) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, experiment_type, json.dumps(cfg), time.time(), notes),
            )

    def finish_experiment(self, run_id: str, total_rounds: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE experiments SET finished_at=?, total_rounds=? WHERE run_id=?",
                (time.time(), total_rounds, run_id),
            )

    # ── Reads ─────────────────────────────────────────────────────────────────

    def get_rounds(
        self,
        run_id: Optional[str] = None,
        attacker_id: Optional[str] = None,
        defender_id: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict]:
        clauses, params = [], []
        if run_id:       clauses.append("run_id=?");      params.append(run_id)
        if attacker_id:  clauses.append("attacker_id=?"); params.append(attacker_id)
        if defender_id:  clauses.append("defender_id=?"); params.append(defender_id)
        if category:     clauses.append("seed_category=?"); params.append(category)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM rounds {where} ORDER BY timestamp DESC LIMIT ?", params
            ).fetchall()
        return [dict(r) for r in rows]

    def get_attack_success_matrix(
        self, run_id: Optional[str] = None, include_counts: bool = False
    ) -> dict:
        """Return {attacker_id: {defender_id: success_rate}} for heatmap.

        With include_counts=True each cell is {"success_rate": float, "total": int}
        instead, so callers can compute confidence intervals.
        """
        where = "WHERE run_id=?" if run_id else ""
        params = [run_id] if run_id else []
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT attacker_id, defender_id,
                       AVG(judge_attack_success) AS success_rate,
                       COUNT(*) AS total
                    FROM rounds {where}
                    GROUP BY attacker_id, defender_id""",
                params,
            ).fetchall()
        matrix: dict[str, dict] = {}
        for row in rows:
            cell = (
                {"success_rate": row["success_rate"], "total": row["total"]}
                if include_counts else row["success_rate"]
            )
            matrix.setdefault(row["attacker_id"], {})[row["defender_id"]] = cell
        return matrix

    def get_category_breakdown(self, run_id: Optional[str] = None) -> list[dict]:
        where = "WHERE run_id=?" if run_id else ""
        params = [run_id] if run_id else []
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT seed_category, attacker_id, defender_id,
                       AVG(judge_attack_success)    AS attack_success_rate,
                       AVG(judge_harm_severity)     AS avg_severity,
                       AVG(judge_defender_correct)  AS defender_accuracy,
                       COUNT(*) AS total
                    FROM rounds {where}
                    GROUP BY seed_category, attacker_id, defender_id
                    ORDER BY attack_success_rate DESC""",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_elo_trajectories(self, run_id: Optional[str] = None) -> list[dict]:
        where = "WHERE run_id=?" if run_id else ""
        params = [run_id] if run_id else []
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM elo_snapshots {where} ORDER BY timestamp",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_summary_stats(self, run_id: Optional[str] = None) -> dict:
        where = "WHERE run_id=?" if run_id else ""
        params = [run_id] if run_id else []
        with self._conn() as conn:
            row = conn.execute(
                f"""SELECT
                    COUNT(*) AS total_rounds,
                    AVG(judge_attack_success)   AS overall_attack_success_rate,
                    AVG(judge_defender_correct) AS overall_defender_accuracy,
                    AVG(judge_harm_severity)    AS avg_severity,
                    AVG(judge_confidence)       AS avg_judge_confidence
                FROM rounds {where}""",
                params,
            ).fetchone()
        return dict(row) if row else {}

    def get_all_run_ids(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT run_id, experiment_type, started_at, total_rounds FROM experiments ORDER BY started_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]
