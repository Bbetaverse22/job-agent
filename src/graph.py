"""Graph wiring v2:

    plan_search → discover → score →(conditional)→ tailor → approval_gate → track
                      ↑                    |
                      └── reformulate ←────┘   (bounded loop, off by default)

The approval_gate uses interrupt(); the graph checkpoints to SQLite and resumes
when a decision payload arrives (Telegram bot or CLI).

Observability: set LANGSMITH_TRACING=true + LANGSMITH_API_KEY and every node,
LLM call, and the loop appear as a trace automatically — no code changes.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import PipelineState
from . import nodes, config


def build_graph(checkpointer=None):
    g = StateGraph(PipelineState)
    g.add_node("plan_search", nodes.plan_search)
    g.add_node("discover", nodes.discover)
    g.add_node("score", nodes.score)
    g.add_node("reformulate", nodes.reformulate)
    g.add_node("tailor", nodes.tailor)
    g.add_node("approval_gate", nodes.approval_gate)
    g.add_node("track", nodes.track)

    g.add_edge(START, "plan_search")
    g.add_edge("plan_search", "discover")
    g.add_edge("discover", "score")
    g.add_conditional_edges("score", nodes.should_reformulate,
                            {"reformulate": "reformulate", "tailor": "tailor"})
    g.add_edge("reformulate", "discover")   # the loop closes here
    g.add_edge("tailor", "approval_gate")
    g.add_edge("approval_gate", "track")
    g.add_edge("track", END)
    return g.compile(checkpointer=checkpointer)


def default_checkpointer():
    import sqlite3
    conn = sqlite3.connect(str(config.ROOT / "checkpoints.db"), check_same_thread=False)
    return SqliteSaver(conn)
