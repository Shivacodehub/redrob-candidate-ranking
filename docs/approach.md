# Approach & design decisions

## The core problem with keyword-based ranking

Traditional ATS filters ask: "does the resume contain the word 'Python'?"  
A great recruiter asks: "has this person *actually used* Python to build things at the right level of complexity, in a trajectory that makes sense for where this role is going?"

Those are completely different questions. Our system answers the second one.

---

## Architecture decisions

### Why a two-stage funnel (embedding + LLM)?

Running a deep LLM evaluation on 100,000+ candidates would cost ~40 hours of T4 time. That's not a pipeline — that's a fire.

The embedding pre-filter handles the full pool in minutes. It's allowed to be imperfect (high recall, lower precision) because the LLM re-ranker corrects its ordering mistakes. The funnel shape — millions → thousands → hundreds → LLM → shortlist — is how every production retrieval system works, from Google to enterprise search. We adopted the same shape.

### Why Qwen2.5-7B over GPT-4/Claude?

Three reasons:
1. **Reproducibility** — the graders can run the exact same weights and get the same output. An API-based system depends on a service that may change or rate-limit.
2. **Cost** — zero API cost means we can evaluate 400 candidates without worrying about spend.
3. **Offline-first** — the system runs fully on-premise, which matters for enterprise hiring data (privacy).

### Why BGE-small over a larger embedding model?

`bge-small-en-v1.5` (33M params) scores within 2–3 points of `bge-large-en-v1.5` (335M) on BEIR benchmarks, runs 10× faster, and uses a fraction of the GPU memory. The pre-filter step only needs to get the top-400 right — it doesn't need to perfectly rank all 100k. Small is the right choice here.

### Why repeat sections in the searchable text blob?

Embedding models compute a single fixed-length vector for an entire document. Sections that appear multiple times in the input effectively receive more "attention weight" in the encoding. By repeating the headline and skills (the two highest signal fields) we push the embedding closer to what a recruiter would consider the most important parts of the profile — without any model fine-tuning.

### Why MMR (Maximal Marginal Relevance) on the final shortlist?

A naive top-K would return the ten most similar candidates — who often look nearly identical (e.g. ten mid-level Python backend engineers at similar companies). A recruiter actually wants a shortlist that covers the search space: one strong generalist, one deep specialist, one with domain experience in an adjacent field. MMR gives that by penalizing candidates who are too similar to already-selected ones.

### Why evidence-weighted skill scoring?

Resume fraud is real. A candidate who lists "Kubernetes" in a skills section but whose entire career history describes marketing work is a red flag. By cross-referencing `skills[]` against `career_history[].description` text, we give 1.5× weight to skills that appear evidenced in work history. This signal feeds directly into the LLM's `authenticity_flag` output.

---

## Scoring formula

```
final_rank_score = 
    0.70 × composite_fit_score (LLM holistic judgment)
  + 0.20 × semantic_similarity × 100 (embedding retrieval sanity-check)
  + 0.10 × engagement_score (recruiter responsiveness, open-to-work, profile completeness)
```

Penalties:
- `major_concern` authenticity flag: × 0.7
- `minor_concern` authenticity flag: × 0.9

The 70/20/10 split is intentional: the LLM's judgment dominates because it can reason across dimensions that embeddings can't (e.g. "this candidate has Python experience but only in a data science context, not backend systems as the JD requires"). The embedding score is a sanity-check, not the primary signal.

---

## What the LLM is asked to do (and not do)

The deep evaluation prompt explicitly asks for:
- `skill_match_score` — not "does the resume mention the skill" but "does the career evidence support depth in this skill at the required level"
- `experience_relevance_score` — relevance of the work history to the specific domain, not just total years
- `trajectory_score` — career growth pattern (is this person trending up? stable? declining?)
- `authenticity_flag` — inconsistencies between claims and evidence
- `reasoning` — 2–3 sentences a recruiter could read, defend, and act on

The LLM is explicitly instructed to be honest and critical rather than flattering — scores should be calibrated to the role requirements, not to make every candidate look good.

---

## Limitations & known gaps

- **No vision/document parsing** — if candidate profiles had attached PDF resumes, we'd need a document extraction step before profiling.
- **Language bias** — BGE-small is English-optimized. Profiles written in regional languages may score lower than deserved.
- **The pre-filter can miss edge cases** — a candidate with highly transferable skills from a different industry may not use the right vocabulary in their profile, causing the embedding stage to miss them. The `PRE_FILTER_TOP_K` knob should be set generously (400+) to reduce this risk.
- **LLM hallucination on scoring** — Qwen2.5-7B occasionally outputs slightly out-of-range numbers or inconsistent JSON. The `extract_json` function and error fallback in the evaluation loop handle this gracefully.
