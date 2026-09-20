"""Telegram human-in-the-loop gate.

The pipeline pauses at approval_gate; this bot sends each pending application
with Approve / Skip inline buttons, collects answers, and resumes the graph.

    python -m src.telegram_gate --thread <thread_id>
Uses long polling — no webhook or public URL needed.
"""
import argparse, json, time, urllib.parse, urllib.request
from langgraph.types import Command
from . import config
from .graph import build_graph, default_checkpointer

API = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}"

def tg(method, **params):
    data = urllib.parse.urlencode({k: v if isinstance(v, str) else json.dumps(v)
                                   for k, v in params.items()}).encode()
    with urllib.request.urlopen(f"{API}/{method}", data=data, timeout=35) as r:
        return json.load(r)

def send_review_request(idx, item):
    notes = f"\n⚠️ {item['notes']}" if item.get("notes") else ""
    text = (f"*{item['company']}* — {item['title']}\n"
            f"Verdict: {item['verdict']} ({item['score']}/100){notes}\n\n"
            f"_Cover letter preview:_\n{item['cover_letter_preview']}")
    kb = {"inline_keyboard": [[
        {"text": "✅ Approve", "callback_data": f"approved:{idx}"},
        {"text": "⏭ Skip", "callback_data": f"skipped:{idx}"}]]}
    tg("sendMessage", chat_id=config.TELEGRAM_CHAT_ID, text=text,
       parse_mode="Markdown", reply_markup=kb)

def collect_decisions(items):
    decisions, offset = {}, 0
    for i, item in enumerate(items):
        send_review_request(i, item)
    while len(decisions) < len(items):
        upd = tg("getUpdates", offset=offset, timeout=30,
                 allowed_updates=["callback_query"])
        for u in upd.get("result", []):
            offset = u["update_id"] + 1
            cq = u.get("callback_query")
            if not cq:
                continue
            decision, idx = cq["data"].split(":")
            idx = int(idx)
            if idx < len(items):
                decisions[items[idx]["url"]] = decision
                tg("answerCallbackQuery", callback_query_id=cq["id"],
                   text=f"{decision}: {items[idx]['company']}")
        time.sleep(1)
    return decisions

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--thread", required=True)
    args = p.parse_args()
    graph = build_graph(checkpointer=default_checkpointer())
    cfg = {"configurable": {"thread_id": args.thread}}
    snap = graph.get_state(cfg)
    interrupts = [t for t in snap.tasks if t.interrupts]
    if not interrupts:
        print("No pending approval interrupt on this thread.")
        return
    payload = interrupts[0].interrupts[0].value
    decisions = collect_decisions(payload["items"])
    print(f"Resuming with: {decisions}")
    for _ in graph.stream(Command(resume=decisions), cfg, stream_mode="values"):
        pass
    print("Pipeline complete. Approved drafts exported to out/.")

if __name__ == "__main__":
    main()
