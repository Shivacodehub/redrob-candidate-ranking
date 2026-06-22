# GitHub setup guide

Run these commands from inside the `redrob-candidate-ranking/` folder.

---

## First-time: Initialize and push to GitHub

```bash
# 1. Go into the project folder
cd redrob-candidate-ranking

# 2. Initialize git (if not already done)
git init

# 3. Add all files (dataset and model weights are gitignored automatically)
git add .

# 4. First commit
git commit -m "Initial commit: candidate ranking pipeline"

# 5. Create a new repo on GitHub:
#    → github.com → New repository → Name: redrob-candidate-ranking
#    → Visibility: Public (required for hackathon submission) or Private
#    → Do NOT initialize with README (we already have one)

# 6. Connect your local repo to GitHub (replace YOUR_USERNAME):
git remote add origin https://github.com/YOUR_USERNAME/redrob-candidate-ranking.git

# 7. Push
git branch -M main
git push -u origin main
```

---

## What will be on GitHub (and what won't)

**Will be pushed:**
- `candidate_ranking_pipeline.ipynb` ✅
- `README.md` ✅
- `requirements.txt` ✅
- `setup_local_gpu.sh` ✅
- `scripts/validate_output.py` ✅
- `docs/approach.md` ✅
- `docs/kaggle_quickstart.md` ✅
- `.gitignore` ✅

**Will NOT be pushed (gitignored):**
- `data/candidates.jsonl` — 4.75 GB, keep it local
- `output/ranked_shortlist.csv` — your submission file, keep it local
- `venv/` — Python environment, recreated by setup script
- Model weights — downloaded at runtime

---

## After making changes: push updates

```bash
git add .
git commit -m "Describe what you changed"
git push
```

---

## If you want to include a sample output in the repo

```bash
# Add just the sample (not the full output)
git add output/ranked_shortlist_sample.csv
git commit -m "Add sample ranked output (10 candidates)"
git push
```
