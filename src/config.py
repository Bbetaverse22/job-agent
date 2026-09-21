"""Configuration and LLM provider.

Observability: LangSmith traces automatically when these env vars are set —
no code changes needed (their "one line" beat, except here it's zero lines):
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=...
    LANGSMITH_PROJECT=job-agent
"""
import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
load_dotenv(ENV_FILE)
# python-dotenv reads `KEY=   # a comment` as the value "# a comment" when nothing
# comes before the comment. That turned blank settings into garbage: MODEL_ID
# became a model name OpenAI rejects, and JSEARCH_API_KEY was sent as a key.
# Treat any value from .env that starts with "#" as unset, both for our settings
# and for the SDKs that read their keys straight from the environment.
for _key, _value in dotenv_values(ENV_FILE).items():
    if _value is not None and _value.strip().startswith("#"):
        os.environ.pop(_key, None)
PROFILE_DIR = ROOT / "profile"
DB_PATH = ROOT / "pipeline.db"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
# `or` guards against CI passing empty strings (an unset GitHub secret expands
# to "" which beats the getenv default and crashes int("")).
SALARY_FLOOR = int(os.getenv("SALARY_FLOOR_USD") or "0")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "bedrock")  # bedrock | anthropic | openai
MODEL_ID = os.getenv("MODEL_ID", "")
# OpenAI reasoning effort: none | low | medium | high. Blank uses the model default.
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "").strip().lower()

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
# Title terms you never want to see, whole-word and case-insensitive, e.g.
# "staff,principal,director,machine learning engineer". Personal taste, so it
# lives here rather than in the built-in non-engineering list.
EXCLUDE_TITLE_TERMS = tuple(
    t.strip().lower() for t in os.getenv("EXCLUDE_TITLE_TERMS", "").split(",") if t.strip())
# Cost guard: at most this many LLM scoring calls per run; overflow is deferred.
MAX_LLM_SCORES = int(os.getenv("MAX_LLM_SCORES_PER_RUN") or "25")


def load_profile() -> str:
    """Concatenate your profile markdown files as grounding context.

    The committed *.example.md templates are skipped. The setup step copies each
    template to its real name, so both exist side by side; loading both would
    feed the model [FILL] placeholders next to your real profile, and the
    matched_skills audit would check claims against template text."""
    parts = []
    for f in sorted(PROFILE_DIR.glob("*.md")):
        if f.name.endswith(".example.md"):
            continue
        parts.append(f"<file name='{f.name}'>\n{f.read_text()}\n</file>")
    return "\n\n".join(parts)


def get_llm(temperature: float = 0.2):
    """Lazy-import the chat model so the graph compiles without cloud creds."""
    if LLM_PROVIDER == "bedrock":
        from langchain_aws import ChatBedrockConverse
        return ChatBedrockConverse(model=MODEL_ID or "us.anthropic.claude-sonnet-4-6-v1:0",
                                   temperature=temperature)
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=MODEL_ID or "claude-sonnet-4-6", temperature=temperature,
                             max_tokens=4096)
    if LLM_PROVIDER == "openai":
        # The Responses API, not Chat Completions: GPT-5.6 models refuse function
        # tools with reasoning on /v1/chat/completions, and structured output is
        # a function tool, so every scoring call failed with a 400 there.
        # No temperature: these reasoning models reject anything but the default.
        from langchain_openai import ChatOpenAI
        extra = {"reasoning": {"effort": OPENAI_REASONING_EFFORT}} if OPENAI_REASONING_EFFORT else {}
        return ChatOpenAI(model=MODEL_ID or "gpt-5.6-luna", use_responses_api=True, **extra)
    raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
                     "Use one of: bedrock, anthropic, openai")


def get_structured_llm(schema, temperature: float = 0.2):
    """A chat model bound to a pydantic schema.

    OpenAI goes through function calling rather than its json_schema mode:
    FitScore uses ge/le bounds, which strict JSON schema does not accept, and
    function calling is the path every provider here handles the same way."""
    llm = get_llm(temperature)
    if LLM_PROVIDER == "openai":
        return llm.with_structured_output(schema, method="function_calling")
    return llm.with_structured_output(schema)
