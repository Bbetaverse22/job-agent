"""job-radar adapter (github.com/maccydee/job-radar).

job-radar owns discovery and application tracking; this adapter feeds its roles
into the pipeline's discover node so scoring/tailoring/Telegram approval run on
top of it. Three ways in, first configured wins:

1. JOBRADAR_JSON — a file produced by `job-radar list --json` (or --new --json)
2. JOBRADAR_DB   — direct read-only read of data/job-radar.db; the schema is
                   introspected at runtime (find the table that has url+title
                   columns) so this survives upstream schema changes
3. JOBRADAR_CMD  — run the CLI yourself, e.g. "job-radar list --new --json"

Like every other source: NEVER raises. Errors log and return [].
Roles job-radar considers settled (rejected/withdrawn/closed/skipped) are
excluded, so its tracking decisions are respected.
"""
import json
import shlex
import sqlite3
import subprocess

from .state import JobPosting
from . import config

SETTLED = {"rejected", "withdrawn", "closed", "skip", "skipped"}

# Flexible key mapping — job-radar's exact field names may differ across
# versions, so we probe a list of candidates per field.
KEYS = {
    "url": ["url", "link", "apply_url", "job_url", "href"],
    "title": ["title", "role", "job_title", "name"],
    "company": ["company", "org", "employer", "organisation", "organization"],
    "location": ["location", "loc", "workplace"],
    "description": ["description", "desc", "content", "text"],
    "salary": ["salary", "salary_text", "pay", "compensation"],
    "posted": ["posted", "posted_at", "date", "first_seen", "created_at"],
    "status": ["status", "state", "application_status"],
}


def _pick(row: dict, field: str) -> str:
    for k in KEYS[field]:
        if k in row and row[k] is not None:
            return str(row[k])
    return ""


def _rows_to_postings(rows: list[dict]) -> list[JobPosting]:
    out = []
    for row in rows:
        url = _pick(row, "url")
        if not url:
            continue
        if _pick(row, "status").lower() in SETTLED:
            continue
        out.append(JobPosting(
            url=url,
            title=_pick(row, "title") or "(untitled)",
            company=_pick(row, "company"),
            source="job-radar",
            location=_pick(row, "location"),
            description=_pick(row, "description")[:20000],
            salary_text=_pick(row, "salary"),
            posted=_pick(row, "posted")))
    return out


def _from_json_text(text: str) -> list[JobPosting]:
    data = json.loads(text)
    if isinstance(data, dict):  # maybe wrapped: {"jobs": [...]} / {"roles": [...]}
        for key in ("jobs", "roles", "results", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    return _rows_to_postings([r for r in data if isinstance(r, dict)])


def _from_db(path: str) -> list[JobPosting]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        best = None
        for t in tables:
            cols = {c[1].lower() for c in con.execute(f'PRAGMA table_info("{t}")')}
            if (cols & set(KEYS["url"])) and (cols & set(KEYS["title"])):
                best = t
                break
        if not best:
            print(f"[job-radar] no table with url+title columns in {path} (tables: {tables})")
            return []
        rows = [dict(r) for r in con.execute(f'SELECT * FROM "{best}"')]
        return _rows_to_postings(rows)
    finally:
        con.close()


def fetch_jobradar() -> list[JobPosting]:
    try:
        if config.JOBRADAR_JSON:
            with open(config.JOBRADAR_JSON) as f:
                return _from_json_text(f.read())
        if config.JOBRADAR_DB:
            return _from_db(config.JOBRADAR_DB)
        if config.JOBRADAR_CMD:
            proc = subprocess.run(shlex.split(config.JOBRADAR_CMD),
                                  capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                print(f"[job-radar] cmd failed: {proc.stderr[:300]}")
                return []
            return _from_json_text(proc.stdout)
    except Exception as e:  # noqa: BLE001 — adapter must never kill a run
        print(f"[job-radar] adapter error: {e}")
    return []
