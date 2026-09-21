"""Shared state for the job application pipeline graph."""
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field
import operator


class JobPosting(BaseModel):
    url: str
    company: str
    title: str
    source: str = ""
    location: str = ""
    description: str = ""
    salary_text: str = ""
    posted: str = ""


class SearchPlan(BaseModel):
    """Model-chosen search arguments (Job Scout's fetch_jobs pattern):
    the LLM reads the profile and picks the query; code executes it."""
    query: str = Field(description="Job search query, e.g. 'AI engineer LLM LangGraph'")
    remote: bool = True
    location: str = "United States"
    rationale: str = ""


class FitScore(BaseModel):
    """Structured output from the scoring agent."""
    hard_fail: bool = Field(description=(
        "True ONLY if a hard constraint fails: not fully remote and not in the candidate's "
        "commutable metro, requires relocation, a STATED salary ceiling below the floor, or "
        "work authorization. Nothing else is a hard fail. A missing skill or language (even "
        "a stated must-have), years of experience, seniority level, and occasional travel "
        "are NEVER hard fails; lower stack_overlap or seniority_fit instead."))
    hard_fail_reason: str = ""
    stack_overlap: int = Field(ge=0, le=30, description="Exact technology matches between the candidate profile and the posting. Count only tools named in both; near-misses and adjacent ecosystems do not count")
    ai_mandate: int = Field(ge=0, le=25, description="25=building LLM/agentic systems IS the job; 10=AI-adjacent; 0=no AI mandate")
    seniority_fit: int = Field(ge=0, le=15)
    company_signal: int = Field(ge=0, le=15)
    process_cost: int = Field(ge=0, le=15, description="Lower for unpaid take-homes, 5+ rounds, weak contract-to-hire terms")
    matched_skills: list[str] = Field(default_factory=list, description="Skills that appear in BOTH the candidate profile and the job text — never claim a skill absent from either")
    rationale: str = ""
    talking_points: list[str] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return self.stack_overlap + self.ai_mandate + self.seniority_fit + self.company_signal + self.process_cost

    @property
    def verdict(self) -> str:
        if self.hard_fail:
            return "REJECT"
        t = self.total
        return "APPLY" if t >= 75 else "APPLY_IF_CAPACITY" if t >= 55 else "HOLD" if t >= 35 else "REJECT"


class ApplicationDraft(BaseModel):
    resume_md: str = ""
    cover_letter_md: str = ""
    review_notes: str = ""


class JobItem(BaseModel):
    """One job flowing through the pipeline."""
    posting: JobPosting
    score: Optional[FitScore] = None
    draft: Optional[ApplicationDraft] = None
    decision: Literal["pending", "approved", "edited", "skipped"] = "pending"
    status: str = "discovered"
    notes: str = ""   # e.g. "unconfirmed salary" — surfaced, never used to hide a role


def merge_jobs(left: list[JobItem], right: list[JobItem]) -> list[JobItem]:
    """Reducer for the jobs channel: merge by posting URL rather than append.

    Nodes that change a job (score, tailor, approval_gate) must RETURN the
    changed items — in-place mutation alone does not survive checkpointing, and
    a plain append reducer would duplicate every job it touched."""
    index = {j.posting.url: i for i, j in enumerate(left)}
    out = list(left)
    for j in right:
        i = index.get(j.posting.url)
        if i is None:
            index[j.posting.url] = len(out)
            out.append(j)
        else:
            out[i] = j
    return out


class PipelineState(BaseModel):
    keywords: list[str] = Field(default_factory=lambda: ["ai", "llm", "machine learning", "agent"])
    plan: Optional[SearchPlan] = None
    reformulation_count: int = 0
    jobs: Annotated[list[JobItem], merge_jobs] = Field(default_factory=list)
    log: Annotated[list[str], operator.add] = Field(default_factory=list)
