"""Configuration and LLM provider.

Observability: LangSmith traces automatically when these env vars are set —
no code changes needed (their "one line" beat, except here it's zero lines):
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=...
    LANGSMITH_PROJECT=job-agent
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
PROFILE_DIR = ROOT / "profile"
DB_PATH = ROOT / "pipeline.db"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
# `or` guards against CI passing empty strings (an unset GitHub secret expands
# to "" which beats the getenv default and crashes int("")).
SALARY_FLOOR = int(os.getenv("SALARY_FLOOR_USD") or "0")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "bedrock")  # bedrock | anthropic
MODEL_ID = os.getenv("MODEL_ID", "")

# Aggregator sources (all optional — cascade degrades gracefully)
JSEARCH_API_KEY = os.getenv("JSEARCH_API_KEY", "")
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "")

# job-radar adapter (github.com/maccydee/job-radar) — three ways in, first match wins:
#   JOBRADAR_JSON  path to a `job-radar list --json` export file
#   JOBRADAR_DB    path to job-radar's data/job-radar.db (read-only, schema introspected)
#   JOBRADAR_CMD   command to run, e.g. "job-radar list --new --json"
JOBRADAR_JSON = os.getenv("JOBRADAR_JSON", "")
JOBRADAR_DB = os.getenv("JOBRADAR_DB", "")
JOBRADAR_CMD = os.getenv("JOBRADAR_CMD", "")

# Reformulation loop (their baseline showed it fires often and roughly triples
# cost/latency for marginal gain — so: off by default, capped at 1)
ENABLE_REFORMULATION = os.getenv("ENABLE_REFORMULATION", "false").lower() == "true"
MAX_REFORMULATIONS = int(os.getenv("MAX_REFORMULATIONS") or "1")
MIN_GOOD_MATCHES = int(os.getenv("MIN_GOOD_MATCHES") or "3")

CASCADE_LIMIT = int(os.getenv("CASCADE_LIMIT") or "25")

# Your own constraints. Kept out of the source so the repo carries no personal
# data. HOME_METRO is the one commutable city, if any, that a non-remote posting
# may still pass on. RESUME_MUST_KEEP is a comma-separated list of landmarks
# (employers, a title, a credential) that must survive resume tailoring; leave it
# empty and that half of the lossiness check simply does not run.
HOME_METRO = os.getenv("HOME_METRO", "").strip().lower()
RESUME_MUST_KEEP = tuple(
    m.strip() for m in os.getenv("RESUME_MUST_KEEP", "").split(",") if m.strip())
# Cost guard: at most this many LLM scoring calls per run; overflow is deferred.
MAX_LLM_SCORES = int(os.getenv("MAX_LLM_SCORES_PER_RUN") or "25")


def load_profile() -> str:
    """Concatenate all profile markdown files as grounding context."""
    parts = []
    for f in sorted(PROFILE_DIR.glob("*.md")):
        parts.append(f"<file name='{f.name}'>\n{f.read_text()}\n</file>")
    return "\n\n".join(parts)


def get_llm(temperature: float = 0.2):
    """Lazy-import the chat model so the graph compiles without cloud creds."""
    if LLM_PROVIDER == "bedrock":
        from langchain_aws import ChatBedrockConverse
        return ChatBedrockConverse(model=MODEL_ID or "us.anthropic.claude-sonnet-4-6-v1:0",
                                   temperature=temperature)
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=MODEL_ID or "claude-sonnet-4-6", temperature=temperature,
                         max_tokens=4096)
