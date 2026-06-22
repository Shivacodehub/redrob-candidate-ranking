#!/usr/bin/env python3
"""
validate_output.py
Checks ranked_shortlist.csv against the submission spec.
Usage: python scripts/validate_output.py output/ranked_shortlist.csv
"""

import sys
import csv
from pathlib import Path


REQUIRED_COLUMNS = [
    "shortlist_rank",
    "candidate_id",
    "final_rank_score",
    "composite_fit_score",
    "skill_match_score",
    "experience_relevance_score",
    "trajectory_score",
    "authenticity_flag",
    "reasoning",
    "key_strengths",
    "key_gaps",
]

VALID_AUTH_FLAGS = {"none", "minor_concern", "major_concern", "error"}
SCORE_COLUMNS = [
    "final_rank_score",
    "composite_fit_score",
    "skill_match_score",
    "experience_relevance_score",
    "trajectory_score",
]


def validate(csv_path: str) -> bool:
    path = Path(csv_path)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return False

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("ERROR: CSV is empty.")
        return False

    # Check columns
    missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        return False
    print(f"OK: All {len(REQUIRED_COLUMNS)} required columns present.")

    errors = []
    for i, row in enumerate(rows, start=2):  # row 2 = first data row
        rank = row.get("shortlist_rank", "")
        cid = row.get("candidate_id", "")

        # rank must be a positive integer
        try:
            assert int(rank) >= 1
        except (ValueError, AssertionError):
            errors.append(f"Row {i}: invalid shortlist_rank '{rank}'")

        # candidate_id must be non-empty
        if not cid.strip():
            errors.append(f"Row {i}: empty candidate_id")

        # scores must be 0-100
        for col in SCORE_COLUMNS:
            val = row.get(col, "")
            try:
                v = float(val)
                if not (0.0 <= v <= 100.0):
                    errors.append(f"Row {i}: {col}={v} out of range [0,100]")
            except ValueError:
                if val:  # None/empty is ok (some candidates fail LLM eval)
                    errors.append(f"Row {i}: {col}='{val}' is not a number")

        # authenticity_flag
        flag = row.get("authenticity_flag", "").strip()
        if flag and flag not in VALID_AUTH_FLAGS:
            errors.append(f"Row {i}: unknown authenticity_flag '{flag}'")

        # reasoning must not be empty
        if not row.get("reasoning", "").strip():
            errors.append(f"Row {i}: empty reasoning field")

    if errors:
        print(f"\nWARNINGS/ERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors)-20} more")
    else:
        print("OK: All rows pass validation.")

    print(f"\nSummary: {len(rows)} candidates in shortlist.")

    # Check ranks are consecutive starting from 1
    ranks = [int(r["shortlist_rank"]) for r in rows if r.get("shortlist_rank", "").isdigit()]
    if sorted(ranks) == list(range(1, len(ranks) + 1)):
        print("OK: Ranks are consecutive 1 to N.")
    else:
        print("WARNING: Ranks are not consecutive 1..N — check for duplicates or gaps.")

    return len(errors) == 0


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "output/ranked_shortlist.csv"
    ok = validate(csv_path)
    sys.exit(0 if ok else 1)
