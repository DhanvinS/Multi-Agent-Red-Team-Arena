"""
Reusable Plotly chart builders for the arena dashboard.
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional


def elo_trajectory_chart(trajectories: list[dict], title: str = "Elo Rating Over Time") -> go.Figure:
    """Line chart of Elo ratings over time for all entities."""
    if not trajectories:
        return go.Figure().update_layout(title=title, template="plotly_dark")

    df = pd.DataFrame(trajectories)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    fig = px.line(
        df, x="timestamp", y="rating", color="entity_id",
        line_dash="entity_type",
        title=title,
        labels={"rating": "Elo Rating", "timestamp": "Time", "entity_id": "Agent"},
        template="plotly_dark",
    )
    fig.update_layout(hovermode="x unified", legend_title="Agent")
    return fig


def attack_success_heatmap(matrix: dict, title: str = "Attack Success Rate Heatmap") -> go.Figure:
    """Heatmap: rows=attackers, cols=defenders, values=success_rate."""
    if not matrix:
        return go.Figure().update_layout(title=title, template="plotly_dark")

    attackers = sorted(matrix.keys())
    defenders = sorted({d for v in matrix.values() for d in v.keys()})

    z = []
    for atk in attackers:
        row = [matrix.get(atk, {}).get(dfn, 0.0) for dfn in defenders]
        z.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z, x=defenders, y=attackers,
        colorscale="RdYlGn_r",
        zmin=0, zmax=1,
        text=[[f"{v:.0%}" for v in row] for row in z],
        texttemplate="%{text}",
        hoverongaps=False,
        colorbar=dict(title="Success Rate"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Defender",
        yaxis_title="Attacker",
        template="plotly_dark",
    )
    return fig


def category_bar_chart(breakdown: list[dict], metric: str = "attack_success_rate",
                        title: Optional[str] = None) -> go.Figure:
    """Grouped bar chart of a metric across categories and defenders."""
    if not breakdown:
        return go.Figure()

    df = pd.DataFrame(breakdown)
    df[metric] = df[metric].fillna(0).round(3)
    title = title or f"{metric.replace('_', ' ').title()} by Category"

    fig = px.bar(
        df, x="seed_category", y=metric, color="defender_id",
        barmode="group",
        title=title,
        labels={
            "seed_category": "Harm Category",
            metric:          metric.replace("_", " ").title(),
            "defender_id":   "Defender",
        },
        template="plotly_dark",
    )
    fig.update_layout(xaxis_tickangle=-35)
    return fig


def elo_leaderboard_bar(leaderboard: list[dict], entity_type: str = "attacker") -> go.Figure:
    if not leaderboard:
        return go.Figure()

    df = pd.DataFrame(leaderboard).sort_values("rating", ascending=True)
    color = "orangered" if entity_type == "attacker" else "mediumseagreen"

    fig = go.Figure(go.Bar(
        x=df["rating"],
        y=df["display_name"],
        orientation="h",
        marker_color=color,
        text=df["rating"].round(0),
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{'Attacker' if entity_type == 'attacker' else 'Defender'} Elo Ratings",
        xaxis_title="Elo Rating",
        yaxis_title="",
        template="plotly_dark",
        showlegend=False,
    )
    return fig


def severity_distribution(rounds: list[dict]) -> go.Figure:
    """Pie chart of harm severity distribution in logged rounds."""
    from collections import Counter
    labels = {1: "Minimal", 2: "Low", 3: "Medium", 4: "High", 5: "Critical"}
    counts = Counter(r.get("judge_harm_severity", 1) for r in rounds)
    fig = go.Figure(go.Pie(
        labels=[labels.get(k, str(k)) for k in sorted(counts.keys())],
        values=[counts[k] for k in sorted(counts.keys())],
        hole=0.4,
        marker_colors=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"],
    ))
    fig.update_layout(title="Harm Severity Distribution", template="plotly_dark")
    return fig


def attacker_win_rate_over_time(rounds: list[dict], attacker_id: str) -> go.Figure:
    """Rolling win rate for a single attacker over successive rounds."""
    import numpy as np
    atk_rounds = [r for r in rounds if r.get("attacker_id") == attacker_id]
    if not atk_rounds:
        return go.Figure()

    atk_rounds.sort(key=lambda x: x.get("round_number", 0))
    successes = [r["judge_attack_success"] for r in atk_rounds]
    rolling = pd.Series(successes).rolling(window=10, min_periods=1).mean()

    fig = go.Figure(go.Scatter(
        y=rolling,
        mode="lines+markers",
        name=f"{attacker_id} rolling win rate",
        line=dict(color="orangered"),
    ))
    fig.update_layout(
        title=f"Rolling Win Rate: {attacker_id}",
        xaxis_title="Round",
        yaxis_title="Win Rate (10-round rolling avg)",
        yaxis_range=[0, 1],
        template="plotly_dark",
    )
    return fig
