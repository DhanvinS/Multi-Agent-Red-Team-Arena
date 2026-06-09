# Multi-Agent Red Team Arena

An AI safety evaluation framework that pits attacker agents against defender agents in a tournament loop, scored by an LLM judge and rated with a chess-style Elo system.

Built as a portfolio project demonstrating red-teaming methodology, multi-agent orchestration, and safety evaluation pipelines.

---

## What it does

**Attacker agents** generate adversarial prompts (jailbreaks, social engineering, roleplay exploits, etc.) against a seed bank of 120 harmful-behavior prompts across 8 categories.

**Defender agents** classify each prompt as safe or harmful using keyword rules, OpenAI Moderation API, or Meta's PromptGuard-86M model.

**A judge agent** (Claude or GPT-4o-mini) evaluates whether the attack succeeded, rates harm severity (1–5), and determines whether the defender was correct.

**Elo ratings** update after every round — attackers and defenders both have live ratings, so you can track which attack strategies and defenses are strongest over time.

Results are stored in SQLite and visualized in a Streamlit dashboard.

---

## Architecture

```
agents/
  attacker.py       — TemplateAttacker (14 jailbreak patterns), OllamaAttacker, OpenAIAttacker, AdaptiveAttacker
  defender.py       — KeywordDefender, OpenAIModerationDefender, PromptGuardDefender
  judge.py          — JudgeAgent (Anthropic or OpenAI); heuristic fallback if no API

arena/
  elo.py            — EloSystem: chess-style rating updates per round
  logger.py         — ArenaLogger: SQLite backend (rounds, elo_snapshots, experiments tables)
  controller.py     — ArenaController: round-robin tournament loop with Rich progress

experiments/
  transferability.py      — Do attacks work across multiple defenders, or are they defender-specific?
  category_breakdown.py   — Which harm categories does each defender miss most?
  adaptive_vs_static.py   — Does an attacker that learns from failed attempts outperform static templates?

dashboard/
  app.py            — Streamlit app (5 tabs: Leaderboard, Heatmap, Categories, Round Log, Experiments)
  charts.py         — Plotly chart builders

scripts/
  run_arena.py              — CLI entry point for running tournaments
  run_experiments.py        — Runs all 3 research experiments
  seed_prompts_downloader.py — Downloads more prompts from HarmBench / AdvBench / JailbreakBench

data/
  seed_prompts.json   — 120 seed prompts (15 × 8 harm categories, built-in, no download needed)
  arena.db            — SQLite database (auto-created on first run)

config.py             — All paths, API keys, model names (loaded from .env)
```

---

## Quick start (no API keys needed)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run a tournament

```bash
# Template-only attackers, keyword defender, heuristic judge — 100% free
python scripts/run_arena.py \
  --attackers template_random template_dan template_roleplay template_hypothetical \
  --defenders keyword_baseline \
  --no-judge \
  --rounds 40
```

### 3. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

---

## Full run (with API keys)

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
# edit .env — add OPENAI_API_KEY and/or ANTHROPIC_API_KEY
```

Then run with real defenders and a real judge:

```bash
python scripts/run_arena.py --rounds 100
```

Default config uses:
- Anthropic `claude-haiku-4-5-20251001` as judge (cheapest Claude)
- OpenAI `omni-moderation-latest` as defender (free endpoint, no token cost)
- All 4 template attackers + OpenAI attacker

---

## Research experiments

Three experiments are included, each producing structured output and saving results to the DB:

```bash
# Run all three (free mode)
python scripts/run_experiments.py --no-judge --defenders keyword_baseline --rounds 10

# Single experiment
python scripts/run_experiments.py --experiment transferability --rounds 20
python scripts/run_experiments.py --experiment category --rounds 10
python scripts/run_experiments.py --experiment adaptive --rounds 15
```

| Experiment | Question answered |
|---|---|
| `transferability` | Do attacks generalize across defenders, or only beat specific ones? |
| `category` | Which harm categories are each defender's blind spots? |
| `adaptive` | Does multi-turn adaptive attacking outperform single-shot static templates? |

---

## Available agents

**Attackers**

| ID | Description |
|---|---|
| `template_random` | Random jailbreak template each round |
| `template_dan` | DAN (Do Anything Now) prompt |
| `template_roleplay` | Character roleplay exploit |
| `template_hypothetical` | Hypothetical framing |
| `template_system` | Fake system-prompt injection |
| `template_escalating` | Escalates template strength round by round |
| `ollama_llama3` | LLaMA-3-8B via local Ollama |
| `ollama_qwen` | Qwen-7B via local Ollama |
| `ollama_mistral` | Mistral-7B via local Ollama |
| `ollama_phi3` | Phi-3 via local Ollama |
| `openai_gpt4omini` | GPT-4o-mini (requires `OPENAI_API_KEY`) |
| `adaptive_*` | Multi-turn adaptive wrapper of any attacker above |

**Defenders**

| ID | Description |
|---|---|
| `keyword_baseline` | Regex keyword rules — intentionally weak baseline |
| `openai_moderation` | OpenAI Moderation API (free endpoint) |
| `promptguard` | Meta PromptGuard-86M via HuggingFace (requires `pip install transformers torch`) |

---

## Adding more seed prompts

```bash
pip install datasets
python scripts/seed_prompts_downloader.py --source all --max 200
```

Sources: HarmBench, AdvBench, JailbreakBench. Deduplicates automatically.

---

## Using Ollama (local LLM attackers)

```bash
# Install Ollama: https://ollama.com
ollama pull llama3:8b

# Then run with local attacker
python scripts/run_arena.py --attackers ollama_llama3 template_random --defenders keyword_baseline --no-judge --rounds 30
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for OpenAI attacker/defender/judge |
| `ANTHROPIC_API_KEY` | — | Required for Anthropic judge |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `JUDGE_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `JUDGE_MODEL` | `claude-haiku-4-5-20251001` | Model ID for judge |
| `DEFAULT_ROUNDS` | `50` | Default rounds per tournament |
| `ELO_K_FACTOR` | `32` | Elo K-factor (higher = faster rating changes) |
| `ELO_INITIAL_RATING` | `1200` | Starting Elo for all agents |
