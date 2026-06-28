# Intelligent Candidate Discovery & Ranking — Redrob AI Challenge

> Ranks candidates the way a great recruiter would — not by matching keywords, but by actually understanding who fits the role.

---
## 🌐 Live Dashboard

Experience the interactive **Neural-X Dashboard** directly in your browser.

<p align="center">
  <a href="https://neural-x.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/🚀%20Launch%20Live%20Dashboard-Visit%20Now-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Dashboard">
  </a>
</p>

> **🔗 Live Demo:** https://neural-x.streamlit.app/
## How it works (30-second version)

```
Job Description ──► Role Intelligence (LLM)  ─┐
                                               ├──► Semantic Pre-filter ──► Deep LLM Scoring ──► Ranked Shortlist
Candidates.jsonl ──► Candidate Profiling ──────┘
```

1. **Role Intelligence** — One LLM call reads the JD and extracts the *real* role: hard vs soft requirements, implicit signals, seniority, a dense search query.  
2. **Candidate Profiling** — Deterministic parsing of all 100k+ candidates into structured features: career trajectory, evidenced vs claimed skills, engagement signals.  
3. **Semantic Pre-filter** — BGE embeddings rank the full pool by cosine similarity in seconds. Only the top-K survive to the expensive step.  
4. **Deep LLM Evaluation** — Qwen2.5-7B scores each shortlisted candidate on skill match, experience relevance, career trajectory, and authenticity.  
5. **MMR Ranking** — Composite score (70% LLM fit + 20% semantic + 10% engagement) + diversity pass so the shortlist isn't ten identical profiles.

---

## Repository layout

```
redrob-candidate-ranking/
├── candidate_ranking_pipeline.ipynb   ← main pipeline (run this)
├── requirements.txt                   ← pip dependencies
├── setup_local_gpu.sh                 ← one-shot setup for your own GPU machine
├── scripts/
│   └── validate_output.py             ← checks your ranked CSV matches submission spec
├── webapp/                            ← (optional) recruiter-facing web UI
│   └── app.py
├── docs/
│   └── approach.md                    ← written explanation of design decisions
├── .gitignore
└── README.md
```

**Dataset files are NOT committed to this repo** (4.75 GB JSONL). See setup instructions below.

---

## Option A — Run on Kaggle (recommended, free 2×T4 GPUs, 30 GB VRAM)

### Step 1 — Upload the dataset to Kaggle

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
2. Upload the entire `India_runs_data_and_ai_challenge/` folder (or just `candidates.jsonl`)
3. Name it something like `redrob-candidates` — note the **dataset slug** shown in the URL (e.g. `yourusername/redrob-candidates`)
4. Set visibility to **Private**

### Step 2 — Create a Kaggle notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook** → **Upload** → upload `candidate_ranking_pipeline.ipynb`
2. In the notebook sidebar → **Data** → **Add data** → search for your private dataset → add it
3. In the **Settings** panel on the right:
   - **Accelerator**: GPU T4 x2
   - **Persistence**: Files only
   - **Internet**: On (needed to download model weights from HuggingFace)

### Step 3 — Fix the one path in the notebook

Open cell 2 (CONFIG) and update:

```python
# Change this to your actual Kaggle dataset slug:
CANDIDATES_PATH = "/kaggle/input/redrob-candidates/candidates.jsonl"

# Paste the job description from job_description.docx:
JOB_DESCRIPTION = """
<paste full text of job_description.docx here>
"""
```

### Step 4 — Run All

**Runtime → Run All** — the notebook will:
- Auto-install missing packages (bitsandbytes, sentence-transformers, accelerate)  
- Load Qwen2.5-7B in 4-bit on GPU 1, BGE embeddings on GPU 0  
- Process all candidates, run deep evaluation, write output files  

Estimated total runtime: **3–5 hours** for 100k candidates with `PRE_FILTER_TOP_K=400`.

### Step 5 — Download outputs

After the run completes, go to **Output** tab on the right → download:
- `output/ranked_shortlist.csv` — your submission file
- `output/ranked_shortlist_full.jsonl` — full reasoning trail

---

## Option B — Run on your own GPU machine

### Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 16 GB single GPU | 2× GPUs (16 GB each) |
| RAM       | 32 GB | 64 GB |
| Storage   | 20 GB free | 50 GB free |
| CUDA      | 11.8+ | 12.1+ |
| Python    | 3.10+ | 3.11 |

### One-shot setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/redrob-candidate-ranking.git
cd redrob-candidate-ranking

# Run setup (creates venv, installs all deps, checks GPU)
bash setup_local_gpu.sh
```

### Place the dataset

```bash
# Create a data/ folder (it's gitignored) and put your files there:
mkdir -p data/
cp /path/to/candidates.jsonl data/
cp /path/to/job_description.docx data/
```

### Configure the notebook

Update cell 2 in the notebook:

```python
CANDIDATES_PATH = "data/candidates.jsonl"   # local path
JOB_DESCRIPTION = """<paste JD text>"""
```

If you have only **one GPU**, change:
```python
EMBED_DEVICE = "cuda:0"
LLM_DEVICE_MAP = "auto"   # both models share the one GPU
```

### Run the notebook

```bash
source venv/bin/activate
jupyter notebook candidate_ranking_pipeline.ipynb
```

Or run headless:
```bash
jupyter nbconvert --to notebook --execute candidate_ranking_pipeline.ipynb \
    --output candidate_ranking_pipeline_executed.ipynb \
    --ExecutePreprocessor.timeout=36000
```

---

## Config knobs

| Variable | Default | Effect |
|----------|---------|--------|
| `PRE_FILTER_TOP_K` | 400 | Candidates sent to LLM scoring. Lower = faster, higher = better recall. |
| `FINAL_SHORTLIST_SIZE` | 25 | Output size. |
| `EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Swap for `bge-large` for better recall at 3× cost. |
| `LLM_MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Swap for `14B` if you have 2×24GB GPUs. |

---

## Output format

`ranked_shortlist.csv` columns:

| Column | Description |
|--------|-------------|
| `shortlist_rank` | 1 = best fit |
| `candidate_id` | Matches input JSONL |
| `final_rank_score` | 0–100 composite |
| `composite_fit_score` | LLM's holistic fit judgment (0–100) |
| `skill_match_score` | Skills vs JD requirements (0–100) |
| `experience_relevance_score` | Work history relevance (0–100) |
| `trajectory_score` | Career growth pattern (0–100) |
| `authenticity_flag` | `none` / `minor_concern` / `major_concern` |
| `reasoning` | 2–3 sentence human-readable explanation |
| `key_strengths` | Semicolon-separated strengths |
| `key_gaps` | Semicolon-separated gaps |

---

## Validate your output

```bash
python scripts/validate_output.py output/ranked_shortlist.csv
```

---

## Tech stack

| Component | Library | Why |
|-----------|---------|-----|
| Embeddings | `sentence-transformers` + `BAAI/bge-small-en-v1.5` | Fast, strong retrieval; fits one T4 |
| LLM reasoning | `transformers` + `Qwen2.5-7B-Instruct` (4-bit NF4) | Strong reasoning, fits second T4 in 4-bit |
| Quantization | `bitsandbytes` | 4-bit NF4 cuts 7B model to ~5 GB VRAM |
| Multi-GPU | `accelerate` + device_map | Embedding on GPU 0, LLM on GPU 1 |
| Diversity | MMR (Maximal Marginal Relevance) | Avoids ten near-identical profiles in shortlist |
| Data | `pandas` + streaming JSONL | Handles 475 MB JSONL without OOM |

---

## License

MIT — built for the Redrob × India Runs Data & AI Challenge.
