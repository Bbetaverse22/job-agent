"""Job sources.

Two layers, both feeding the discover node:
1. ATS watchlist (Greenhouse/Lever/Ashby public board APIs) — precise, per-company.
2. Aggregator cascade (JSearch → Adzuna → Remotive) — broad coverage, model-chosen
   query. Every adapter follows one rule borrowed from Job Scout: NEVER raise.
   On any error it returns an empty list and the cascade falls through.
No LinkedIn scraping — ToS prohibits automation.
"""
import json
import urllib.parse
import urllib.request
from pathlib import Path

from .state import JobPosting
from . import config

UA = {"User-Agent": "job-agent/1.0"}


def _get(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


# ---------- layer 1: ATS watchlist ----------

def greenhouse(slug: str):
    data = _get(f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true&pay_transparency=true")
    for j in data.get("jobs", []):
        yield JobPosting(url=j["absolute_url"], company=slug, title=j["title"],
                         source="greenhouse",
                         location=(j.get("location") or {}).get("name", ""),
                         description=j.get("content", "")[:20000],
                         salary_text=str(j.get("pay_input_ranges") or ""),
                         posted=str(j.get("updated_at", "")))


def lever(slug: str):
    for j in _get(f"https://api.lever.co/v0/postings/{slug}?mode=json"):
        yield JobPosting(url=j["hostedUrl"], company=slug, title=j["text"],
                         source="lever",
                         location=(j.get("categories") or {}).get("location", ""),
                         description=(j.get("descriptionPlain") or "")[:20000],
                         posted=str(j.get("createdAt", "")))


def ashby(slug: str):
    data = _get(f"https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true")
    for j in data.get("jobs", []):
        yield JobPosting(url=j["jobUrl"], company=slug, title=j["title"], source="ashby",
                         location=j.get("location", ""),
                         description=(j.get("descriptionPlain") or "")[:20000],
                         salary_text=str(j.get("compensation", "")),
                         posted=str(j.get("publishedAt", "")))


FETCHERS = {"greenhouse": greenhouse, "lever": lever, "ashby": ashby}


def fetch_watchlist(companies_file: Path, keywords: list[str]) -> list[JobPosting]:
    out, kws = [], [k.lower() for k in keywords]
    if not companies_file.exists():
        return out
    for line in companies_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        ats, slug = (s.strip() for s in line.split(":", 1))
        fetch = FETCHERS.get(ats.lower())
        if not fetch:
            continue
        try:
            for posting in fetch(slug):
                text = f"{posting.title} {posting.description[:2000]}".lower()
                if not kws or any(k in text for k in kws):
                    out.append(posting)
        except Exception as e:  # noqa: BLE001 — one bad board shouldn't kill the run
            print(f"[discover] watchlist error {ats}:{slug}: {e}")
    return out


# ---------- layer 2: aggregator cascade (never raise) ----------

def jsearch(query: str, remote: bool) -> list[JobPosting]:
    if not config.JSEARCH_API_KEY:
        return []
    try:
        q = urllib.parse.quote(f"{query} remote" if remote else query)
        data = _get(f"https://jsearch.p.rapidapi.com/search?query={q}&num_pages=1&country=us",
                    headers={"X-RapidAPI-Key": config.JSEARCH_API_KEY,
                             "X-RapidAPI-Host": "jsearch.p.rapidapi.com"})
        return [JobPosting(
            url=j.get("job_apply_link") or j.get("job_url", ""),
            company=j.get("employer_name", ""), title=j.get("job_title", ""),
            source="jsearch",
            location=f'{j.get("job_city") or ""} {j.get("job_state") or ""} '
                     f'{"remote" if j.get("job_is_remote") else ""}'.strip(),
            description=(j.get("job_description") or "")[:20000],
            salary_text=str(j.get("job_salary") or ""),
            posted=str(j.get("job_posted_at_datetime_utc") or ""))
            for j in data.get("data", []) if j.get("job_apply_link") or j.get("job_url")]
    except Exception as e:  # noqa: BLE001
        print(f"[discover] jsearch error: {e}")
        return []


def adzuna(query: str) -> list[JobPosting]:
    if not (config.ADZUNA_APP_ID and config.ADZUNA_APP_KEY):
        return []
    try:
        q = urllib.parse.quote(query)
        data = _get("https://api.adzuna.com/v1/api/jobs/us/search/1"
                    f"?app_id={config.ADZUNA_APP_ID}&app_key={config.ADZUNA_APP_KEY}"
                    f"&what={q}&results_per_page=20")
        return [JobPosting(
            url=j.get("redirect_url", ""), title=j.get("title", ""),
            company=(j.get("company") or {}).get("display_name", ""),
            source="adzuna",
            location=(j.get("location") or {}).get("display_name", ""),
            description=(j.get("description") or "")[:20000],
            salary_text=str(j.get("salary_min") or ""))
            for j in data.get("results", []) if j.get("redirect_url")]
    except Exception as e:  # noqa: BLE001
        print(f"[discover] adzuna error: {e}")
        return []


def remotive(query: str) -> list[JobPosting]:
    """Keyless remote-jobs API — the cascade's no-credentials fallback."""
    try:
        q = urllib.parse.quote(query)
        data = _get(f"https://remotive.com/api/remote-jobs?search={q}&limit=20")
        return [JobPosting(
            url=j.get("url", ""), title=j.get("title", ""),
            company=j.get("company_name", ""), source="remotive",
            location=j.get("candidate_required_location", "remote"),
            description=(j.get("description") or "")[:20000],
            salary_text=str(j.get("salary") or ""),
            posted=str(j.get("publication_date") or ""))
            for j in data.get("jobs", []) if j.get("url")]
    except Exception as e:  # noqa: BLE001
        print(f"[discover] remotive error: {e}")
        return []


def fetch_cascade(query: str, remote: bool = True,
                  limit: int | None = None) -> tuple[list[JobPosting], list[str]]:
    """Try sources in order until we have enough. Returns (jobs, sources_used)."""
    limit = limit or config.CASCADE_LIMIT
    jobs: list[JobPosting] = []
    sources: list[str] = []

    def add(name: str, items: list[JobPosting]):
        if items:
            sources.append(name)
            jobs.extend(items)

    add("jsearch", jsearch(query, remote))
    if len(jobs) < 5:
        add("adzuna", adzuna(query))
    if remote or len(jobs) < 5:
        add("remotive", remotive(query))

    seen, deduped = set(), []
    for j in jobs:
        if j.url not in seen:
            seen.add(j.url)
            deduped.append(j)
    return deduped[:limit], sources
