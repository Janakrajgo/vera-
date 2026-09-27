"""
Configuration for the Vera AI Bot.
All secrets and tunable parameters loaded from environment variables.
"""
import os
from pathlib import Path

# ── LLM Provider ──────────────────────────────────────────────────────────
# Supported: "gemini", "openai", "anthropic"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# OpenAI-compatible
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# ── Composition Settings ──────────────────────────────────────────────────
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
COMPOSE_TIMEOUT_SEC = int(os.getenv("COMPOSE_TIMEOUT_SEC", "25"))

# ── Server ────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))

# ── Team Metadata ─────────────────────────────────────────────────────────
TEAM_NAME = os.getenv("TEAM_NAME", "Vera Reborn")
TEAM_MEMBERS = os.getenv("TEAM_MEMBERS", "Janak Raj Goyal").split(",")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "janak@example.com")
BOT_VERSION = "1.0.0"

# ── Dataset Paths ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
EXPANDED_DIR = DATASET_DIR / "expanded"
