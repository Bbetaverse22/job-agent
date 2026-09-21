# job-agent

A LangGraph pipeline that finds job postings, scores them against your profile,
drafts tailored application materials, and stops. **It never submits anything.**
Every application waits for you to approve it, on Telegram or in the terminal,
and approved drafts are written to `out/` for you to send yourself.

```
plan_search → discover → score →(cond.)→ tailor → approval_gate → track
                 ↑                  |
                 └── reformulate ←──┘   (bounded loop, off by default)
```

## What each node does

**plan_search** reads `profile/*.md` and picks the search query. If the model is
unavailable or returns nothing usable, a deterministic fallback query takes over,
because an agent that hard-fails on a skipped tool call cannot run unattended.

**discover** pulls from three layers: a [job-radar](https://github.com/maccydee/job-radar)
export, an ATS watchlist (Greenhouse, Lever, Ashby board APIs, from
`companies.txt`), and an aggregator cascade (JSearch, then Adzuna, then Remotive).
Every adapter follows one rule: never raise. On any error it logs, returns an
empty list, and the cascade falls through. Remotive needs no key, so discovery
works with zero credentials configured. Results are deduplicated against the
tracker, with job-radar winning ties. There is no LinkedIn scraping; their terms
prohibit automation.

**score** runs a deterministic filter first, so no posting earns a paid model call
until it has passed remote/location rules, a non-engineering title check, and the
salary rule. What survives goes to a structured-output rubric. `matched_skills` is
then audited in code: any skill the model claimed that does not literally appear in
both your profile and the job text is dropped, and the drop is recorded in the
rationale. Scoring is capped at `MAX_LLM_SCORES_PER_RUN`; the overflow is deferred
rather than dropped, and re-queued on the next run.

**reformulate** broadens the query when too few strong matches come back. It is
capped at one iteration and disabled by default, because in testing the loop roughly
tripled cost and latency for marginal gain.

**tailor** writes the resume and cover letter grounded only in `profile/*.md`. The
resume is an edit of `profile/00-base-resume.md`, not a rewrite: the model may reword
the summary and reorder bullets, nothing else. A lossiness check counts bullets and
looks for the landmarks in `RESUME_MUST_KEEP`, and falls back to shipping your own
resume unchanged if the tailored version dropped content.

**approval_gate** calls `interrupt()`. The graph checkpoints to SQLite and resumes
when a decision arrives, so you can start a run on your laptop and approve from your
phone hours later.

**track** writes to SQLite, optionally mirrors to Supabase, and exports approved
drafts to `out/<company-role>/`.

## The salary rule

Only a **stated** salary below your floor disqualifies a role. An unstated salary is
flagged `unconfirmed salary` and surfaced everywhere the role appears: the score log,
the Telegram card, the CLI review, `main.py status`, and the exported posting. Most
employers publish no figure, so treating "unstated" as a failure throws away half the
board.

Day and hourly rates are annualized before comparison, so $600/day reads as $156k/yr
rather than $600.

## Prompt injection

Job descriptions are third-party text fetched from hundreds of servers, and they flow
straight into the scoring and tailoring prompts. They are treated as hostile input:
fenced between markers, with those markers stripped from the text first so a posting
cannot close the fence and escape into the instruction context. Both prompts state
that fenced content is claims about a job, never instructions, and that anything
resembling an instruction should be reported in the rationale.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env                                              # then fill it in
for f in profile/*.example.md; do cp "$f" "${f%.example.md}.md"; done
```

Complete the `[FILL]`s in `profile/*.md` and put real board slugs in `companies.txt`.

`profile/00-base-resume.md` is the one that matters most. It is the resume the
pipeline ships, and the lossiness check falls back to it. Without it the run still
works, but that guard is inactive and `tailor` logs a warning.

## Usage

```bash
python main.py discover        # sources + deterministic filter. No LLM, no credentials
python main.py run             # the full pipeline, pausing at the approval gate
python -m src.telegram_gate --thread X    # approve from your phone
python main.py approve --cli --thread X   # or in the terminal
python main.py status          # the tracker
```

`discover` is the one to start with. It skips `plan_search` entirely, takes its query
from `--query` or the fallback, runs the sources, applies the deterministic filter,
and prints what passed plus a count per rejection reason. It costs nothing and needs
no API key.

```bash
python main.py discover --query "AI engineer LangGraph remote"
python main.py discover --keywords ai,llm,agent --all
```

Rows land at `status='discovered'`, which the dedupe deliberately does not treat as
seen, so a later full `run` re-queues them for scoring.

## Model providers

Set `LLM_PROVIDER` to `openai`, `anthropic`, or `bedrock`, and override the model
with `MODEL_ID`. An unrecognized provider fails at startup rather than silently
falling back to another one.

A full run scores up to `MAX_LLM_SCORES_PER_RUN` postings and tailors the few that
pass, so cost is small but not zero. On a cheap tier such as `gpt-5.6-luna` a run
costs cents. Lower `MAX_LLM_SCORES_PER_RUN` while you are testing.

## Observability

Set `LANGSMITH_TRACING=true` with an API key and every node, model call, and loop
iteration is traced. No code changes.

## Configuration

Everything personal lives in `.env` and `profile/`, never in source. `HOME_METRO` is
the one commutable city a non-remote posting may still pass on, blank for fully
remote only. `RESUME_MUST_KEEP` is the list of landmarks that must survive tailoring.
See `.env.example` for the rest.

## Privacy

`profile/*.md`, `.env`, `out/`, and the `.db` files are gitignored. They hold your
resume material, your credentials, and your application history. The committed
`*.example.md` templates show the structure without the content.

If you fork this, keep it that way.
