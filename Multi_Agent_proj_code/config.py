import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODELS = {
    "llama3": "llama3:8b",
    "qwen":   "qwen:7b",
    "mistral": "mistral:7b",
    "phi3":   "phi3:mini",
}

# ── Judge ─────────────────────────────────────────────────────────────────────
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "anthropic")
JUDGE_MODEL    = os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001")

# ── Arena ─────────────────────────────────────────────────────────────────────
DEFAULT_ROUNDS      = int(os.getenv("DEFAULT_ROUNDS", "50"))
ELO_K_FACTOR        = float(os.getenv("ELO_K_FACTOR", "32"))
ELO_INITIAL_RATING  = float(os.getenv("ELO_INITIAL_RATING", "1200.0"))

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
DATA_DIR           = os.path.join(BASE_DIR, "data")
DB_PATH            = os.path.join(DATA_DIR, "arena.db")
SEED_PROMPTS_PATH  = os.path.join(DATA_DIR, "seed_prompts.json")

# ── Harm categories ───────────────────────────────────────────────────────────
HARM_CATEGORIES = [
    "jailbreaks",
    "weapons",
    "misinformation",
    "social_engineering",
    "hate_speech",
    "self_harm",
    "illegal_activities",
    "privacy_violation",
]

# ── Severity levels ───────────────────────────────────────────────────────────
SEVERITY_LABELS = {1: "minimal", 2: "low", 3: "medium", 4: "high", 5: "critical"}
