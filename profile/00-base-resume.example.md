# Base Resume
> Copy this to profile/00-base-resume.md and paste your real resume below the
> instruction block. This file is the resume the pipeline ships. Tailoring is an
> EDIT of it, never a rewrite: the model may reword the professional summary and
> reorder bullets, and nothing else.
>
> Lines starting with ">" and this "# Base Resume" heading are stripped before
> the resume is used, so keep notes to yourself in this block.
>
> Two things depend on this file existing:
>   1. the tailor prompt, which is told to reproduce it in full;
>   2. _resume_too_lossy() in src/nodes.py, which falls back to this resume if
>      the tailored version drops roles or bullets.
> Without it the pipeline still runs, but that guard is inactive and tailor logs
> a warning on every run.
>
> After pasting your resume, update RESUME_MUST_KEEP in src/nodes.py to a few
> landmarks that must survive tailoring — employers, a title, a credential.
> The defaults assume the employers and degree below.

## [FILL Your Name]
[FILL email] · [FILL phone] · [FILL city, state] · [FILL LinkedIn] · [FILL GitHub]

## Professional Summary
[FILL 2-3 sentences. This is the ONLY prose the tailor step may reword.]

## Experience

### [FILL Title] — [FILL Employer]  ([FILL start] – [FILL end])
- [FILL bullet with a concrete result and a metric]
- [FILL bullet]
- [FILL bullet]

### [FILL Title] — [FILL Employer]  ([FILL start] – [FILL end])
- [FILL bullet]
- [FILL bullet]

## Education
- [FILL degree] — [FILL institution, year]

## Skills
**[FILL group, e.g. AI/LLM]:** [FILL]
**[FILL group, e.g. Backend]:** [FILL]
**[FILL group, e.g. Data/Infra]:** [FILL]

## Certifications
- [FILL only certifications you actually hold]
