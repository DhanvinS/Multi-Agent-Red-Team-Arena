"""
Multi-Agent Red Team Arena — Streamlit Dashboard

Launch: streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from arena.logger import ArenaLogger
from dashboard.charts import (
    elo_trajectory_chart,
    attack_success_heatmap,
    category_bar_chart,
    elo_leaderboard_bar,
    severity_distribution,
    attacker_win_rate_over_time,
)
import config

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Red Team Arena",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #cba6f7; }
    .metric-label { font-size: 0.8rem; color: #a6adc8; margin-top: 4px; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_resource
def get_logger():
    return ArenaLogger(config.DB_PATH)


def load_data(logger: ArenaLogger, run_id: str | None):
    rounds        = logger.get_rounds(run_id=run_id, limit=5000)
    matrix        = logger.get_attack_success_matrix(run_id)
    breakdown     = logger.get_category_breakdown(run_id)
    trajectories  = logger.get_elo_trajectories(run_id)
    stats         = logger.get_summary_stats(run_id)
    return rounds, matrix, breakdown, trajectories, stats


# ── Sidebar ────────────────────────────────────────────────────────────────────

def sidebar(logger: ArenaLogger) -> str | None:
    st.sidebar.title("🏟️ Red Team Arena")
    st.sidebar.markdown("---")

    runs = logger.get_all_run_ids()
    if not runs:
        st.sidebar.warning("No runs found. Run the arena first:\n`python scripts/run_arena.py`")
        return None

    run_options = {f"{r['run_id']} ({r['experiment_type']}, {r['total_rounds']} rounds)": r["run_id"]
                   for r in runs}
    run_options["All runs (aggregated)"] = None

    selected_label = st.sidebar.selectbox("Select Run", list(run_options.keys()))
    selected_run = run_options[selected_label]

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Links**")
    st.sidebar.markdown("- [GitHub](https://github.com)")
    st.sidebar.markdown("- [HarmBench](https://huggingface.co/datasets/harmbench)")
    st.sidebar.markdown("- [JailbreakBench](https://jailbreakbench.github.io)")

    return selected_run


# ── Metric cards ───────────────────────────────────────────────────────────────

def metric_cards(stats: dict, rounds: list[dict]):
    total   = stats.get("total_rounds") or 0
    asr     = stats.get("overall_attack_success_rate") or 0
    acc     = stats.get("overall_defender_accuracy") or 0
    sev     = stats.get("avg_severity") or 0
    conf    = stats.get("avg_judge_confidence") or 0

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, value, label in [
        (c1, f"{total:,}",          "Total Rounds"),
        (c2, f"{asr:.1%}",          "Attack Success Rate"),
        (c3, f"{acc:.1%}",          "Defender Accuracy"),
        (c4, f"{sev:.2f}/5",        "Avg Harm Severity"),
        (c5, f"{conf:.2f}",         "Avg Judge Confidence"),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Tabs ───────────────────────────────────────────────────────────────────────

def tab_leaderboard(logger: ArenaLogger, run_id):
    st.header("🏆 Elo Leaderboard")

    rounds, matrix, breakdown, trajectories, stats = load_data(logger, run_id)

    # Derive current Elo from latest round data
    atk_elo: dict[str, dict] = {}
    dfn_elo: dict[str, dict] = {}
    for r in rounds:
        aid = r["attacker_id"]
        did = r["defender_id"]
        atk_elo[aid] = {"display_name": aid, "rating": r["attacker_elo_after"],
                        "entity_id": aid}
        dfn_elo[did] = {"display_name": did, "rating": r["defender_elo_after"],
                        "entity_id": did}

    col1, col2 = st.columns(2)
    with col1:
        if atk_elo:
            st.plotly_chart(
                elo_leaderboard_bar(list(atk_elo.values()), "attacker"),
                use_container_width=True,
            )
        else:
            st.info("No attacker data yet.")

    with col2:
        if dfn_elo:
            st.plotly_chart(
                elo_leaderboard_bar(list(dfn_elo.values()), "defender"),
                use_container_width=True,
            )
        else:
            st.info("No defender data yet.")

    # Rating trajectories
    if trajectories:
        st.plotly_chart(elo_trajectory_chart(trajectories), use_container_width=True)
    else:
        st.info("Run with more rounds to see Elo trajectories.")


def tab_heatmap(logger: ArenaLogger, run_id):
    st.header("🗺️ Attack Success Heatmap")
    st.markdown("Each cell shows the fraction of rounds where attacker bypassed defender.")

    _, matrix, _, _, _ = load_data(logger, run_id)
    if matrix:
        st.plotly_chart(attack_success_heatmap(matrix), use_container_width=True)
    else:
        st.info("No matrix data yet.")


def tab_categories(logger: ArenaLogger, run_id):
    st.header("📊 Category Breakdown")
    st.markdown("Attack success rate and defender accuracy broken down by harm category.")

    _, _, breakdown, _, _ = load_data(logger, run_id)

    if not breakdown:
        st.info("No category data yet.")
        return

    metric = st.selectbox(
        "Metric",
        ["attack_success_rate", "defender_accuracy", "avg_severity"],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    st.plotly_chart(category_bar_chart(breakdown, metric), use_container_width=True)

    # Table view
    with st.expander("Raw data table"):
        df = pd.DataFrame(breakdown)
        df = df[["seed_category", "attacker_id", "defender_id",
                 "attack_success_rate", "defender_accuracy", "avg_severity", "total"]]
        df.columns = ["Category", "Attacker", "Defender",
                      "Attack Success", "Defender Accuracy", "Avg Severity", "Rounds"]
        for col in ["Attack Success", "Defender Accuracy"]:
            df[col] = df[col].apply(lambda x: f"{x:.1%}" if x is not None else "N/A")
        st.dataframe(df, use_container_width=True)


def tab_round_log(logger: ArenaLogger, run_id):
    st.header("📋 Round Log")

    rounds, _, _, _, _ = load_data(logger, run_id)
    if not rounds:
        st.info("No rounds logged yet.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    attackers = sorted({r["attacker_id"] for r in rounds})
    defenders = sorted({r["defender_id"] for r in rounds})
    categories = sorted({r["seed_category"] for r in rounds})

    with col1:
        sel_atk = st.multiselect("Attacker", attackers, default=attackers)
    with col2:
        sel_def = st.multiselect("Defender", defenders, default=defenders)
    with col3:
        sel_cat = st.multiselect("Category", categories, default=categories)
    show_success_only = st.checkbox("Show successful attacks only")

    filtered = [
        r for r in rounds
        if r["attacker_id"] in sel_atk
        and r["defender_id"] in sel_def
        and r["seed_category"] in sel_cat
        and (not show_success_only or r["judge_attack_success"] == 1)
    ]

    st.markdown(f"Showing **{len(filtered)}** of {len(rounds)} rounds")

    if filtered:
        # Severity distribution
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.plotly_chart(severity_distribution(filtered), use_container_width=True)
        with col_b:
            # Attacker rolling win rate
            if sel_atk:
                atk_to_chart = sel_atk[0]
                st.plotly_chart(
                    attacker_win_rate_over_time(filtered, atk_to_chart),
                    use_container_width=True,
                )

        # Round table
        display_cols = [
            "round_number", "attacker_id", "defender_id", "seed_category",
            "judge_attack_success", "judge_harm_severity", "defender_flagged",
            "judge_reasoning",
        ]
        df = pd.DataFrame(filtered)[display_cols]
        df.columns = ["Round", "Attacker", "Defender", "Category",
                      "Atk Win", "Severity", "Def Flagged", "Reasoning"]
        df["Atk Win"] = df["Atk Win"].map({1: "✅", 0: "❌"})
        df["Def Flagged"] = df["Def Flagged"].map({1: "🚩", 0: "—"})
        st.dataframe(df, use_container_width=True, height=400)

        # Detail expander
        st.markdown("---")
        round_ids = [r["round_number"] for r in filtered[:50]]
        sel_round = st.selectbox("Inspect round", round_ids)
        if sel_round:
            r = next((x for x in filtered if x["round_number"] == sel_round), None)
            if r:
                st.markdown(f"**Attack Prompt**")
                st.code(r["attack_prompt"], language=None)
                st.markdown(f"**Judge Reasoning:** {r['judge_reasoning']}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Policy Violation", "Yes" if r["judge_policy_violation"] else "No")
                col2.metric("Harm Severity", f"{r['judge_harm_severity']}/5")
                col3.metric("Defender Confidence", f"{r['defender_confidence']:.2f}")


def tab_experiments(logger: ArenaLogger, run_id):
    st.header("🔬 Experiments")
    runs = logger.get_all_run_ids()
    if not runs:
        st.info("No experiments recorded yet.")
        return

    df = pd.DataFrame(runs)
    df["started_at"] = pd.to_datetime(df["started_at"], unit="s")
    df.columns = ["Run ID", "Type", "Config", "Started", "Total Rounds"]
    st.dataframe(df[["Run ID", "Type", "Started", "Total Rounds"]], use_container_width=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    logger = get_logger()
    run_id = sidebar(logger)

    st.title("🏟️ Multi-Agent Red Team Arena")
    st.markdown(
        "An automated safety evaluation platform — attackers generate adversarial prompts, "
        "defenders classify them, and a judge scores every exchange via Elo."
    )
    st.markdown("---")

    rounds, matrix, breakdown, trajectories, stats = load_data(logger, run_id)

    if not rounds:
        st.warning(
            "No data yet. Run the arena to populate results:\n\n"
            "```bash\npython scripts/run_arena.py --rounds 50\n```"
        )
        return

    metric_cards(stats, rounds)
    st.markdown("---")

    tabs = st.tabs(["🏆 Leaderboard", "🗺️ Heatmap", "📊 Categories", "📋 Round Log", "🔬 Experiments"])

    with tabs[0]: tab_leaderboard(logger, run_id)
    with tabs[1]: tab_heatmap(logger, run_id)
    with tabs[2]: tab_categories(logger, run_id)
    with tabs[3]: tab_round_log(logger, run_id)
    with tabs[4]: tab_experiments(logger, run_id)


if __name__ == "__main__":
    main()
