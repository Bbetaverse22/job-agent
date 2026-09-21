"""Run the pipeline.

    python main.py run                # plan → discover → score → tailor, pause at approval
    python main.py discover           # sources + deterministic filter only — NO LLM, no credentials
    python main.py approve --cli --thread X
    python main.py status
"""
import argparse, sqlite3, sys, uuid
from langgraph.types import Command
from src import config
from src.graph import build_graph, default_checkpointer
from src.state import PipelineState, SearchPlan

def run():
    thread = uuid.uuid4().hex[:8]
    graph = build_graph(checkpointer=default_checkpointer())
    cfg = {"configurable": {"thread_id": thread}}
    for step in graph.stream(PipelineState(), cfg, stream_mode="values"):
        for line in step.get("log", [])[-3:]:
            print(line)
    snap = graph.get_state(cfg)
    if any(t.interrupts for t in snap.tasks):
        print(f"\nPaused for approval. thread_id={thread}")
        print(f"  Telegram: python -m src.telegram_gate --thread {thread}")
        print(f"  Local:    python main.py approve --cli --thread {thread}")
    else:
        print("\nRun complete — nothing reached the approval gate.")

def discover_only(query="", keywords="", show_rejected=False):
    """Discovery + the deterministic hard filter. No LLM call, so this runs with
    no ANTHROPIC_API_KEY / Bedrock credentials and costs nothing.

    plan_search is skipped entirely (it is the first LLM call) — the query comes
    from --query or the deterministic FALLBACK_PLAN. Rows land in the tracker at
    status='discovered', which _tracked_urls() deliberately does not treat as
    seen, so a later full `run` re-queues them for scoring."""
    from src import nodes

    plan = (SearchPlan(query=query, rationale="--query override") if query
            else nodes.FALLBACK_PLAN)
    state = PipelineState(plan=plan)
    if keywords:
        state.keywords = [k.strip().lower() for k in keywords.split(",") if k.strip()]

    print(f"watchlist: {nodes.COMPANIES_FILE.name}")
    print(f"query:    '{plan.query}'")
    print(f"keywords: {state.keywords or '(none — watchlist unfiltered)'}\n")

    result = nodes.discover(state)
    for line in result["log"]:
        print(line)

    passed, rejected = [], []
    for item in result["jobs"]:
        reason = nodes._hard_filter(item)
        (rejected if reason else passed).append((item, reason))

    print(f"\n=== {len(passed)} passed the hard filter ===")
    for item, _ in sorted(passed, key=lambda pair: nodes._scoring_priority(pair[0])):
        note = f"  [{item.notes}]" if item.notes else ""
        print(f"  {item.posting.company} — {item.posting.title}")
        print(f"     {item.posting.location or '(no location)'} · {item.posting.source}{note}")
        print(f"     {item.posting.url}")

    if rejected:
        by_reason = {}
        for item, reason in rejected:
            by_reason.setdefault(reason, []).append(item)
        print(f"\n=== {len(rejected)} rejected by the hard filter ===")
        for reason, items in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            print(f"  {len(items):>3}  {reason}")
            if show_rejected:
                for item in items:
                    print(f"         {item.posting.company} — {item.posting.title}")
        if not show_rejected:
            print("  (--all to list them)")

    print(f"\nNothing was scored, drafted, or submitted. "
          f"Run `python main.py run` with credentials to score these.")


def approve_cli(thread):
    graph = build_graph(checkpointer=default_checkpointer())
    cfg = {"configurable": {"thread_id": thread}}
    snap = graph.get_state(cfg)
    tasks = [t for t in snap.tasks if t.interrupts]
    if not tasks:
        print("No pending approvals.")
        return
    items = tasks[0].interrupts[0].value["items"]
    decisions = {}
    for it in items:
        note = f"  ⚠️ {it['notes']}" if it.get("notes") else ""
        print(f"\n{it['company']} — {it['title']}  [{it['verdict']} {it['score']}/100]{note}")
        print(it["cover_letter_preview"], "\n")
        ans = input("approve / skip? [a/s] ").strip().lower()
        decisions[it["url"]] = "approved" if ans.startswith("a") else "skipped"
    for _ in graph.stream(Command(resume=decisions), cfg, stream_mode="values"):
        pass
    print("Done. Approved drafts in out/.")

def status():
    """Print the tracker. A run that has never completed discovery leaves no
    applications table, so report that rather than raising OperationalError."""
    con = sqlite3.connect(config.DB_PATH)
    try:
        exists = con.execute("SELECT name FROM sqlite_master WHERE type='table' "
                             "AND name='applications'").fetchone()
        if not exists:
            print("No applications tracked yet — run `python main.py run` first.")
            return
        rows = con.execute("SELECT company,title,status,verdict,score,decision,notes,last_update "
                           "FROM applications ORDER BY last_update DESC LIMIT 30").fetchall()
        if not rows:
            print("No applications tracked yet — run `python main.py run` first.")
        for r in rows:
            print(" | ".join(str(x) for x in r))
    finally:
        con.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "discover", "approve", "status"])
    p.add_argument("--cli", action="store_true")
    p.add_argument("--thread", default="")
    p.add_argument("--query", default="", help="discover: override the search query")
    p.add_argument("--keywords", default="",
                   help="discover: comma-separated watchlist filter (default: ai,llm,...)")
    p.add_argument("--all", action="store_true",
                   help="discover: also list the postings the hard filter rejected")
    a = p.parse_args()
    if a.cmd == "run":
        run()
    elif a.cmd == "discover":
        discover_only(a.query, a.keywords, a.all)
    elif a.cmd == "approve":
        if not a.thread:
            sys.exit("--thread required")
        approve_cli(a.thread)
    else:
        status()
