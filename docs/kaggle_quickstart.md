# Kaggle quick-start guide

Step-by-step for running this on Kaggle's free 2×T4 GPUs.

---

## 1. Upload the dataset to Kaggle (do this once)

1. Download the challenge zip: `India_runs_data_and_ai_challenge.zip`
2. Extract it — you should see `candidates.jsonl`, `job_description.docx`, etc.
3. Go to → https://www.kaggle.com/datasets/new
4. Click **Upload files** → drag `candidates.jsonl` (the big one, ~4.75 GB)
5. Also upload `job_description.docx`, `sample_candidates.json`, `sample_submission.csv`
6. Name: `redrob-challenge-data` (or anything memorable)
7. Visibility: **Private** → click **Create**
8. Note the dataset URL — it will look like:  
   `https://www.kaggle.com/datasets/yourusername/redrob-challenge-data`  
   Your slug is `yourusername/redrob-challenge-data`

---

## 2. Create a Kaggle notebook from this repo

1. Go to → https://www.kaggle.com/code → **New Notebook**
2. Top-right → **File → Upload Notebook** → select `candidate_ranking_pipeline.ipynb`
3. Kaggle will open it in the editor

---

## 3. Attach your dataset

In the right sidebar → **Data** section → click **+ Add Data**  
Search: `redrob-challenge-data` → click **Add**

Your files will now be available at `/kaggle/input/redrob-challenge-data/`

---

## 4. Enable GPU and Internet

Right sidebar → **Session options**:
- **Accelerator** → GPU T4 x2  ← critical
- **Persistence** → Files only  
- **Internet** → On  ← needed to download model weights from HuggingFace

---

## 5. Read the job description from the docx

The JD is in `job_description.docx`. To extract it, add this to a cell before the CONFIG cell:

```python
# One-time: extract text from job_description.docx
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-docx"], check=True)

from docx import Document
doc = Document("/kaggle/input/redrob-challenge-data/job_description.docx")
jd_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
print(jd_text[:2000])  # preview first 2000 chars
```

Then paste the printed text into `JOB_DESCRIPTION` in the CONFIG cell.

---

## 6. Update the CONFIG cell

```python
CANDIDATES_PATH = "/kaggle/input/redrob-challenge-data/candidates.jsonl"

JOB_DESCRIPTION = """
<paste the full text extracted from job_description.docx here>
"""

PRE_FILTER_TOP_K = 400   # safe for 9hr budget
FINAL_SHORTLIST_SIZE = 25
```

---

## 7. Run All

**Run → Run All** or **Shift+Enter** through each cell.

The first time, HuggingFace will download:
- `BAAI/bge-small-en-v1.5` (~130 MB) 
- `Qwen/Qwen2.5-7B-Instruct` (~4 GB in 4-bit)

Download is fast on Kaggle's network (~2–3 min).

---

## 8. Monitor progress

Watch the deep evaluation cell — it prints every 25 candidates:

```
[25/400] elapsed=4.2min | avg=10.1s/candidate | ETA=62.3min remaining
[50/400] elapsed=8.3min | avg=9.9s/candidate | ETA=57.9min remaining
```

If the ETA looks too long, stop the kernel and reduce `PRE_FILTER_TOP_K` to 200.

---

## 9. Download outputs

After completion → **Output** tab (right sidebar):
- `/kaggle/working/output/ranked_shortlist.csv` — **submit this**
- `/kaggle/working/output/ranked_shortlist_full.jsonl` — full reasoning trail

Click the file → **Download**

---

## Troubleshooting

**"CUDA out of memory"**  
→ Reduce `PRE_FILTER_TOP_K` to 200 or lower  
→ Make sure GPU T4 x2 is selected (not x1)  

**"No JSON object found in LLM output"**  
→ Usually transient; the loop catches and skips bad outputs automatically  
→ If it happens for every candidate, the model may be loading incorrectly — restart kernel  

**"Path not found: /kaggle/input/..."**  
→ The dataset isn't attached. Re-attach via the Data sidebar → Add Data  

**Slow download of model weights**  
→ Normal. HuggingFace download happens once per session. Subsequent runs use the cached version in `/root/.cache/huggingface/`  

**"Module not found: bitsandbytes"**  
→ The auto-install cell at the top handles this. Make sure you ran it first.
