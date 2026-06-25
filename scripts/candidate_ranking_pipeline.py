import os, json, re, gc, time, math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

print("Torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} | "
          f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

import os, json, re, gc, time, math
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("Torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} | "
          f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

CANDIDATES_PATH = "data/candidates.jsonl"

JOB_DESCRIPTION = """
Job Description: Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid — flexible cadence) | Open to relocation candidates from Tier-1 Indian cities
Employment Type: Full-time
Experience Required: 5–9 years

What you'd actually be doing:
Own the intelligence layer of Redrob's product — ranking, retrieval, and matching systems.
Weeks 1-3: Audit current BM25 + rule-based scoring. Identify 3-4 highest-leverage fixes.
Weeks 4-8: Ship v2 ranking system with embeddings, hybrid retrieval, LLM-based re-ranking.
Weeks 9-12: Set up evaluation infrastructure — offline benchmarks, online A/B testing, recruiter-feedback loops.

Things you ABSOLUTELY need:
- Production experience with embeddings-based retrieval (sentence-transformers, BGE, E5, OpenAI embeddings) deployed to real users. Must have handled embedding drift, index refresh, retrieval-quality regression in production.
- Production experience with vector databases or hybrid search (Pinecone, Weaviate, Qdrant, Milvus, Elasticsearch, FAISS).
- Strong Python, strong code quality.
- Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation).

Things we'd like but won't reject for:
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank models (XGBoost-based or neural)
- Prior exposure to HR-tech, recruiting tech, or marketplace products
- Distributed systems or large-scale inference optimization
- Open-source contributions in AI/ML

DISQUALIFIERS:
- Pure research background with no production deployment
- AI experience only from recent LangChain/OpenAI wrapper projects (under 12 months)
- Senior engineers who haven't written production code in 18+ months
- Entire career at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant) only
- Primary expertise in computer vision, speech, or robotics without NLP/IR exposure
- Title-chasers switching companies every 1.5 years
- Work entirely on closed-source systems for 5+ years

Ideal candidate:
- 6-8 years total, 4-5 in applied ML/AI at product companies (not pure services)
- Shipped at least one end-to-end ranking, search, or recommendation system at meaningful scale
- Strong opinions on retrieval (hybrid vs dense), evaluation (offline vs online), LLM integration (fine-tune vs prompt)
- Located in or willing to relocate to Noida or Pune
- Active on job market / Redrob platform

Key insight: Do NOT just match keywords. Reason about what the JD means vs what it says.
A candidate who built a recommendation system at a product company fits — even if they don't use words like RAG or Pinecone.
A candidate with all AI keywords but whose title is Marketing Manager does NOT fit.
Weigh behavioral signals: candidates inactive for 6+ months with low recruiter response rates should be down-weighted.
"""

PRE_FILTER_TOP_K = 800
FINAL_SHORTLIST_SIZE = 25
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
EMBED_DEVICE = "cuda:0"
LLM_DEVICE_MAP = {"": 1}
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# Install/upgrade only what's missing on the Kaggle image. Safe to re-run.
import importlib
def _need(pkg):
    try:
        importlib.import_module(pkg)
        return False
    except ImportError:
        return True

to_install = []
if _need("sentence_transformers"): to_install.append("sentence-transformers")
if _need("bitsandbytes"): to_install.append("bitsandbytes")
if _need("accelerate"): to_install.append("accelerate")

if to_install:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *to_install], check=True)
    print("Installed:", to_install)
else:
    print("All required packages already present.")

def load_candidates_jsonl(path: str) -> list:
    path = Path(path)
    if not path.exists():
        print(f"!! Path not found: {path}")
        print("Contents of /kaggle/input:")
        for p in Path("/kaggle/input").glob("**/*"):
            print(" ", p)
        raise FileNotFoundError(f"Fix CANDIDATES_PATH above -- {path} does not exist.")

    records = []
    n_bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                n_bad += 1
                continue

    print(f"Loaded {len(records):,} candidate records ({n_bad} malformed lines skipped)")
    return records

raw_candidates = load_candidates_jsonl(CANDIDATES_PATH)
print(json.dumps(raw_candidates[0], indent=2)[:1500])

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def parse_candidate(rec: dict) -> dict:
    cid = rec.get("candidate_id", "UNKNOWN")
    profile = rec.get("profile", {}) or {}
    history = rec.get("career_history", []) or []
    education = rec.get("education", []) or []
    skills = rec.get("skills", []) or []
    certifications = rec.get("certifications", []) or []
    signals = rec.get("redrob_signals", {}) or {}

    # --- career trajectory -------------------------------------------------
    total_months = sum(h.get("duration_months", 0) or 0 for h in history)
    n_roles = len(history)
    avg_tenure_months = (total_months / n_roles) if n_roles else 0.0
    # crude job-hopping signal: many short roles (<9mo) relative to total roles
    short_stints = sum(1 for h in history if (h.get("duration_months") or 0) < 9)
    job_hopping_ratio = (short_stints / n_roles) if n_roles else 0.0

    # title seniority trend: does the most recent role look senior vs first?
    seniority_kw = ["intern", "junior", "associate", "engineer", "senior",
                     "lead", "principal", "staff", "manager", "director",
                     "head", "vp", "chief"]
    def seniority_rank(title):
        title_l = (title or "").lower()
        for rank, kw in enumerate(seniority_kw):
            if kw in title_l:
                return rank
        return 4  # default mid-level if no keyword matches

    history_sorted = sorted(history, key=lambda h: h.get("start_date") or "")
    if history_sorted:
        seniority_trend = seniority_rank(history_sorted[-1].get("title")) - seniority_rank(history_sorted[0].get("title"))
    else:
        seniority_trend = 0

    # --- evidenced vs claimed skills ----------------------------------------
    history_text_blob = " ".join(
        (h.get("description", "") or "") + " " + (h.get("title", "") or "")
        for h in history
    ).lower()

    skill_names = []
    evidenced_skill_names = []
    proficiency_weight = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
    weighted_skill_score = 0.0
    for s in skills:
        name = (s.get("name") or "").strip()
        if not name:
            continue
        skill_names.append(name)
        prof = proficiency_weight.get(s.get("proficiency", ""), 1)
        endorsements = s.get("endorsements", 0) or 0
        base = prof + min(endorsements / 10, 3)  # cap endorsement contribution
        is_evidenced = name.lower() in history_text_blob
        if is_evidenced:
            evidenced_skill_names.append(name)
            base *= 1.5  # evidenced skills count more
        weighted_skill_score += base

    evidenced_ratio = (len(evidenced_skill_names) / len(skill_names)) if skill_names else 0.0

    # --- education tier ------------------------------------------------------
    tier_rank = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}
    best_tier = max((tier_rank.get(e.get("tier", "unknown"), 0) for e in education), default=0)

    # --- redrob engagement / availability score -------------------------------
    def safe_num(v, default=0.0):
        return v if isinstance(v, (int, float)) else default

    responsiveness = safe_num(signals.get("recruiter_response_rate"), 0.0)
    interview_rate = safe_num(signals.get("interview_completion_rate"), 0.0)
    offer_accept = signals.get("offer_acceptance_rate", -1)
    offer_accept = offer_accept if isinstance(offer_accept, (int, float)) and offer_accept >= 0 else None
    github_score = signals.get("github_activity_score", -1)
    github_score = github_score if isinstance(github_score, (int, float)) and github_score >= 0 else None
    completeness = safe_num(signals.get("profile_completeness_score"), 0.0)
    open_to_work = bool(signals.get("open_to_work_flag", False))
    notice_days = signals.get("notice_period_days", None)

    engagement_score = np.mean([
        responsiveness * 100,
        interview_rate * 100,
        completeness,
        (offer_accept * 100) if offer_accept is not None else 50,  # neutral if unknown
    ])

    # --- searchable text blob for embeddings ----------------------------------
    # repeat high-signal sections so the bi-encoder weights them more than a
    # flat concatenation would (cheap, effective trick for short embedder context)
    text_parts = [
        profile.get("headline", ""),
        profile.get("headline", ""),  # headline repeated: strongest fit signal
        profile.get("current_title", ""),
        profile.get("summary", ""),
        f"{profile.get('years_of_experience', '')} years experience",
        " ".join(skill_names),
        " ".join(skill_names),  # skills repeated: directly matched against JD requirements
        " ".join(f"{h.get('title','')} at {h.get('company','')}: {h.get('description','')}" for h in history[:5]),
    ]
    searchable_text = " . ".join(p for p in text_parts if p).strip()

    return {
        "candidate_id": cid,
        "name": profile.get("anonymized_name", ""),
        "headline": profile.get("headline", ""),
        "current_title": profile.get("current_title", ""),
        "current_company": profile.get("current_company", ""),
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "years_of_experience": profile.get("years_of_experience", 0),
        "total_months_experience": total_months,
        "n_roles": n_roles,
        "avg_tenure_months": round(avg_tenure_months, 1),
        "job_hopping_ratio": round(job_hopping_ratio, 2),
        "seniority_trend": seniority_trend,
        "skill_names": skill_names,
        "evidenced_skill_names": evidenced_skill_names,
        "evidenced_ratio": round(evidenced_ratio, 2),
        "weighted_skill_score": round(weighted_skill_score, 2),
        "best_education_tier": best_tier,
        "engagement_score": round(float(engagement_score), 1),
        "open_to_work": open_to_work,
        "notice_period_days": notice_days,
        "github_activity_score": github_score,
        "offer_acceptance_rate": offer_accept,
        "expected_salary_min_lpa": safe_get(signals, "expected_salary_range_inr_lpa", "min"),
        "expected_salary_max_lpa": safe_get(signals, "expected_salary_range_inr_lpa", "max"),
        "preferred_work_mode": signals.get("preferred_work_mode", "unknown"),
        "willing_to_relocate": bool(signals.get("willing_to_relocate", False)),
        "searchable_text": searchable_text,
        "_raw": rec,  # keep raw record for stage 4 deep-dive prompts
    }


t0 = time.time()
parsed = [parse_candidate(r) for r in raw_candidates]
candidates_df = pd.DataFrame(parsed)
print(f"Parsed {len(candidates_df):,} candidates in {time.time()-t0:.1f}s")
candidates_df.drop(columns=["_raw", "skill_names", "evidenced_skill_names"]).head(3)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading LLM (this can take a few minutes the first time)...")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    quantization_config=bnb_config,
    device_map=LLM_DEVICE_MAP,
    torch_dtype=torch.float16,
)
llm_model.eval()
print("LLM loaded.")

def llm_generate(prompt: str, max_new_tokens: int = 800, temperature: float = 0.2) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer(text, return_tensors="pt").to(llm_model.device)
    with torch.no_grad():
        out = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
            pad_token_id=llm_tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return llm_tokenizer.decode(gen, skip_special_tokens=True).strip()


def extract_json(text: str) -> dict:
    """Pull the first {...} JSON object out of a model response, tolerating
    markdown fences or extra prose around it."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM output:\n{text[:500]}")
    return json.loads(match.group(0))

ROLE_EXTRACTION_PROMPT = """You are a senior technical recruiter analyzing a job description.
Extract a structured breakdown of what this role ACTUALLY needs -- not just keywords,
but the underlying intent. Distinguish hard requirements from nice-to-haves, and infer
implicit needs the JD doesn't state outright (e.g. "fast-paced startup" implies
adaptability and ownership).

Job description:
---
{jd}
---

Respond with ONLY a JSON object in this exact shape, no other text:
{{
  "role_title": "string",
  "seniority_level": "junior|mid|senior|lead|principal|executive",
  "hard_requirements": ["list of non-negotiable requirements, skills, or experience"],
  "soft_requirements": ["list of nice-to-have skills or experience"],
  "implicit_needs": ["list of inferred traits/skills not explicitly stated but implied by context"],
  "min_years_experience": number,
  "key_skills": ["ranked list of the most important technical/functional skills"],
  "domain_context": "string describing industry/domain context",
  "search_query": "a dense natural-language paragraph optimized for semantic search, describing the ideal candidate for this role in 3-4 sentences"
}}
"""

raw_response = llm_generate(ROLE_EXTRACTION_PROMPT.format(jd=JOB_DESCRIPTION), max_new_tokens=900, temperature=0.1)
role_dna = extract_json(raw_response)
print(json.dumps(role_dna, indent=2))

from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)
print(f"Embedding model loaded on {EMBED_DEVICE}")

BATCH_SIZE = 256

t0 = time.time()
candidate_embeddings = embed_model.encode(
    candidates_df["searchable_text"].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
print(f"Embedded {len(candidates_df):,} candidates in {time.time()-t0:.1f}s | shape={candidate_embeddings.shape}")

np.save(OUTPUT_DIR / "candidate_embeddings.npy", candidate_embeddings)

jd_embedding = embed_model.encode(
    [role_dna["search_query"]],
    convert_to_numpy=True,
    normalize_embeddings=True,
)[0]

similarity_scores = candidate_embeddings @ jd_embedding
candidates_df["semantic_similarity"] = similarity_scores

# Optional light hard-filter on minimum experience, if the JD specified one.
min_years = role_dna.get("min_years_experience")
pool = candidates_df.copy()
if isinstance(min_years, (int, float)) and min_years > 0:
    before = len(pool)
    # soft floor: allow a little slack (1 year) since YOE reporting is noisy
    pool = pool[pool["years_of_experience"] >= max(min_years - 1, 0)]
    print(f"Hard filter (min {min_years} yrs experience, with 1yr slack): {before:,} -> {len(pool):,}")

pool = pool.sort_values("semantic_similarity", ascending=False)
prefiltered = pool.head(PRE_FILTER_TOP_K).reset_index(drop=True)
print(f"Pre-filter pool: {len(prefiltered)} candidates carried into deep LLM evaluation")
prefiltered[["candidate_id", "current_title", "years_of_experience", "semantic_similarity"]].head(10)

DEEP_EVAL_PROMPT = """You are an expert technical recruiter scoring how well a candidate
fits a specific role. Be honest and critical -- do not inflate scores. Flag any
inconsistencies between claimed skills and actual career evidence.

ROLE REQUIREMENTS:
{role_dna}

CANDIDATE PROFILE:
- Title: {title} at {company}
- Experience: {years} years ({n_roles} roles, avg tenure {avg_tenure} months)
- Headline: {headline}
- Summary: {summary}
- Skills (claimed): {skills}
- Skills with evidence in work history: {evidenced_skills}
- Recent roles: {recent_roles}
- Education tier (0=unknown, 4=top tier): {edu_tier}
- Platform engagement score (0-100): {engagement}

Respond with ONLY a JSON object in this exact shape, no other text:
{{
  "skill_match_score": number (0-100),
  "experience_relevance_score": number (0-100),
  "trajectory_score": number (0-100, considers growth pattern and stability),
  "authenticity_flag": "none|minor_concern|major_concern",
  "authenticity_notes": "string, brief, only if flag is not none",
  "composite_fit_score": number (0-100, your overall judgment, NOT a simple average),
  "reasoning": "2-3 sentence explanation a recruiter could read and trust",
  "key_strengths": ["short", "phrases"],
  "key_gaps": ["short", "phrases"]
}}
"""

def build_eval_prompt(row, role_dna):
    history = row["_raw"].get("career_history", [])
    recent = "; ".join(
        f"{h.get('title','')} at {h.get('company','')} ({h.get('duration_months',0)}mo)"
        for h in history[:3]
    )
    return DEEP_EVAL_PROMPT.format(
        role_dna=json.dumps({k: role_dna[k] for k in
                              ["role_title", "seniority_level", "hard_requirements",
                               "soft_requirements", "implicit_needs", "key_skills"]}),
        title=row["current_title"], company=row["current_company"],
        years=row["years_of_experience"], n_roles=row["n_roles"],
        avg_tenure=row["avg_tenure_months"], headline=row["headline"],
        summary=row["_raw"].get("profile", {}).get("summary", ""),
        skills=", ".join(row["skill_names"][:20]),
        evidenced_skills=", ".join(row["evidenced_skill_names"][:20]) or "none detected",
        recent_roles=recent, edu_tier=row["best_education_tier"],
        engagement=row["engagement_score"],
    )

# Deep evaluation loop. Prints periodic progress + ETA so you can sanity-check
# the pace early and adjust PRE_FILTER_TOP_K if needed before committing the
# full 9hr budget.

eval_results = []
t0 = time.time()
n = len(prefiltered)

for i, row in prefiltered.iterrows():
    prompt = build_eval_prompt(row, role_dna)
    try:
        response = llm_generate(prompt, max_new_tokens=500, temperature=0.1)
        result = extract_json(response)
    except Exception as e:
        result = {
            "skill_match_score": None, "experience_relevance_score": None,
            "trajectory_score": None, "authenticity_flag": "error",
            "authenticity_notes": str(e)[:200], "composite_fit_score": None,
            "reasoning": "Evaluation failed for this candidate.",
            "key_strengths": [], "key_gaps": [],
        }
    result["candidate_id"] = row["candidate_id"]
    eval_results.append(result)

    if (i + 1) % 25 == 0 or (i + 1) == n:
        elapsed = time.time() - t0
        rate = elapsed / (i + 1)
        eta_min = rate * (n - i - 1) / 60
        print(f"[{i+1}/{n}] elapsed={elapsed/60:.1f}min | "
              f"avg={rate:.2f}s/candidate | ETA={eta_min:.1f}min remaining")

eval_df = pd.DataFrame(eval_results)
print("Deep evaluation complete.")
eval_df.head(5)

merged = prefiltered.merge(eval_df, on="candidate_id", how="left")

# Composite ranking score: 70% LLM fit judgment, 20% semantic similarity
# (sanity-check against pure retrieval), 10% engagement/availability --
# a great-fit candidate who's unreachable is operationally less useful.
merged["composite_fit_score"] = merged["composite_fit_score"].fillna(0)
merged["final_rank_score"] = (
    0.70 * merged["composite_fit_score"]
    + 0.20 * (merged["semantic_similarity"] * 100)
    + 0.10 * merged["engagement_score"]
)

# Penalize major authenticity concerns rather than hard-excluding --
# let the recruiter see them with full context instead of silently dropping.
merged.loc[merged["authenticity_flag"] == "major_concern", "final_rank_score"] *= 0.7
merged.loc[merged["authenticity_flag"] == "minor_concern", "final_rank_score"] *= 0.9

ranked = merged.sort_values("final_rank_score", ascending=False).reset_index(drop=True)
ranked["rank"] = ranked.index + 1


def select_diverse_shortlist(df, k, similarity_matrix, lambda_param=0.7):
    """Simple MMR: greedily pick high-scoring candidates while penalizing
    similarity to already-selected ones, so the shortlist isn't ten
    near-identical profiles."""
    selected_idx = []
    candidate_idx = list(df.index)
    scores = df["final_rank_score"].values
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    while len(selected_idx) < min(k, len(df)):
        best_i, best_val = None, -1e9
        for i in candidate_idx:
            if i in selected_idx:
                continue
            relevance = norm_scores[df.index.get_loc(i)]
            if selected_idx:
                sim_to_selected = max(
                    similarity_matrix[df.index.get_loc(i), df.index.get_loc(j)]
                    for j in selected_idx
                )
            else:
                sim_to_selected = 0
            mmr = lambda_param * relevance - (1 - lambda_param) * sim_to_selected
            if mmr > best_val:
                best_val, best_i = mmr, i
        selected_idx.append(best_i)

    return df.loc[selected_idx]


# Build a similarity matrix restricted to the ranked pool's own embeddings
# (re-uses already-computed embeddings, indexed back from the original df).
ranked_embed_idx = ranked["candidate_id"].map(
    dict(zip(candidates_df["candidate_id"], range(len(candidates_df))))
).values
ranked_embeddings = candidate_embeddings[ranked_embed_idx]
sim_matrix = ranked_embeddings @ ranked_embeddings.T

shortlist = select_diverse_shortlist(ranked, FINAL_SHORTLIST_SIZE, sim_matrix)
shortlist = shortlist.sort_values("final_rank_score", ascending=False).reset_index(drop=True)
shortlist["shortlist_rank"] = shortlist.index + 1

print(f"Final diverse shortlist: {len(shortlist)} candidates")
shortlist[["shortlist_rank", "candidate_id", "current_title", "current_company",
           "final_rank_score", "composite_fit_score", "authenticity_flag"]]

csv_cols = [
    "shortlist_rank", "candidate_id", "name", "current_title", "current_company",
    "location", "years_of_experience", "final_rank_score", "composite_fit_score",
    "skill_match_score", "experience_relevance_score", "trajectory_score",
    "semantic_similarity", "engagement_score", "authenticity_flag",
    "reasoning", "key_strengths", "key_gaps",
]
shortlist_out = shortlist[csv_cols].copy()
shortlist_out["key_strengths"] = shortlist_out["key_strengths"].apply(lambda x: "; ".join(x) if isinstance(x, list) else x)
shortlist_out["key_gaps"] = shortlist_out["key_gaps"].apply(lambda x: "; ".join(x) if isinstance(x, list) else x)

csv_path = OUTPUT_DIR / "ranked_shortlist.csv"
shortlist_out.to_csv(csv_path, index=False)
print(f"Wrote {csv_path}")

jsonl_path = OUTPUT_DIR / "ranked_shortlist_full.jsonl"
with open(jsonl_path, "w") as f:
    for _, row in shortlist.iterrows():
        record = {
            "shortlist_rank": int(row["shortlist_rank"]),
            "candidate_id": row["candidate_id"],
            "name": row["name"],
            "current_title": row["current_title"],
            "current_company": row["current_company"],
            "final_rank_score": round(float(row["final_rank_score"]), 2),
            "scores": {
                "composite_fit": row["composite_fit_score"],
                "skill_match": row["skill_match_score"],
                "experience_relevance": row["experience_relevance_score"],
                "trajectory": row["trajectory_score"],
                "semantic_similarity": round(float(row["semantic_similarity"]), 4),
                "engagement": row["engagement_score"],
            },
            "authenticity_flag": row["authenticity_flag"],
            "authenticity_notes": row["authenticity_notes"],
            "reasoning": row["reasoning"],
            "key_strengths": row["key_strengths"],
            "key_gaps": row["key_gaps"],
        }
        f.write(json.dumps(record) + "\n")
print(f"Wrote {jsonl_path}")

# Also save the full ranked pool (pre-diversity-filter) for audit/debugging.
ranked.drop(columns=["_raw"]).to_csv(OUTPUT_DIR / "full_ranked_pool.csv", index=False)
print(f"Wrote {OUTPUT_DIR / 'full_ranked_pool.csv'} ({len(ranked)} candidates, full pre-filter pool with scores)")

del llm_model
gc.collect()
torch.cuda.empty_cache()
print("GPU memory freed.")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.memory_allocated(i)/1e9:.2f} GB allocated")
