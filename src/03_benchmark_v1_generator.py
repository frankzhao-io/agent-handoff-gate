"""Benchmark v1 Generator (N=48): Clean Controls + 5 Fault-Injection Categories.

Generates 48 randomized multi-step tabular data-analysis tasks in data/benchmark_v1/:
- C0_clean_control (8): Clean CSVs & accurate upstream plan (measures overhead / no-harm)
- C1_whitespace_col (8): Hidden spaces in column names + corrupted ASSUMED_SCHEMA
- C2_currency_na (8): Currency formatting & 'N/A' strings + corrupted ASSUMED_DTYPE
- C3_dirty_category (8): Unnormalized casing/spaces in keys + corrupted ASSUMED_VALUES
- C4_mixed_dates (8): Mixed date formats & invalid tokens + corrupted ASSUMED_FORMAT
- C5_join_key_mismatch (8): Int vs zero-padded string join keys + corrupted ASSUMED_JOIN_KEY
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "benchmark_v1"


def generate_benchmark_v1(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks: list[dict] = []

    # C0: Clean Control (8 tasks)
    for i in range(8):
        tid = f"C0_clean_{i:02d}"
        tdir = OUT_DIR / tid
        tdir.mkdir(exist_ok=True)
        groups = ["Alpha", "Beta", "Gamma"]
        rows = [{"group": rng.choice(groups), "metric": float(rng.randint(10, 100))} for _ in range(12)]
        rows[0]["group"] = "Alpha"
        df = pd.DataFrame(rows)
        fname = f"clean_{i}.csv"
        df.to_csv(tdir / fname, index=False)
        ans = float(df[df["group"] == "Alpha"]["metric"].sum())
        tasks.append({
            "task_id": tid,
            "category": "C0_clean_control",
            "is_corrupted_plan": False,
            "work_dir": str(tdir),
            "files": [fname],
            "question": f"Compute the sum of 'metric' where group == 'Alpha' in {fname}. Print only the final float.",
            "expected_answer": ans,
            "tolerance": 1e-4,
            "upstream_plan": {
                "steps": [f"Load {fname}", "Filter group == 'Alpha'", "Sum 'metric'"],
                "assumed_claims": [],
            },
        })

    # C1: Whitespace in column names (8 tasks)
    for i in range(8):
        tid = f"C1_ws_{i:02d}"
        tdir = OUT_DIR / tid
        tdir.mkdir(exist_ok=True)
        col_raw = rng.choice([" revenue ", " profit", "score "])
        col_clean = col_raw.strip()
        segs = ["North", "South", "East"]
        rows = [{"segment": rng.choice(segs), col_raw: float(rng.randint(20, 200))} for _ in range(10)]
        rows[0]["segment"] = "North"
        df = pd.DataFrame(rows)
        fname = f"ledger_{i}.csv"
        df.to_csv(tdir / fname, index=False)
        ans = float(df[df["segment"] == "North"][col_raw].sum())
        tasks.append({
            "task_id": tid,
            "category": "C1_whitespace_col",
            "is_corrupted_plan": True,
            "work_dir": str(tdir),
            "files": [fname],
            "question": f"Compute the total {col_clean} for segment == 'North' in {fname}. Print only the final float.",
            "expected_answer": ans,
            "tolerance": 1e-4,
            "upstream_plan": {
                "steps": [f"Load {fname}", "Filter segment == 'North'", f"Sum column '{col_clean}'"],
                "assumed_claims": [f"ASSUMED_SCHEMA: {fname} has exact columns ['segment', '{col_clean}'] with float64 dtype."],
            },
        })

    # C2: Currency strings + invalid tokens (8 tasks)
    for i in range(8):
        tid = f"C2_curr_{i:02d}"
        tdir = OUT_DIR / tid
        tdir.mkdir(exist_ok=True)
        vals = [rng.randint(100, 2500) for _ in range(6)]
        na_tok = rng.choice(["N/A", "missing", "--"])
        raw_col = [f"${v:,}.00" for v in vals] + [na_tok, na_tok]
        rng.shuffle(raw_col)
        df = pd.DataFrame({"tx_id": list(range(len(raw_col))), "amount": raw_col})
        fname = f"payments_{i}.csv"
        df.to_csv(tdir / fname, index=False)
        ans = float(sum(vals) / len(vals))
        tasks.append({
            "task_id": tid,
            "category": "C2_currency_na",
            "is_corrupted_plan": True,
            "work_dir": str(tdir),
            "files": [fname],
            "question": f"Compute the mean of valid numeric values in 'amount' of {fname}. Print only the final float.",
            "expected_answer": ans,
            "tolerance": 1e-3,
            "upstream_plan": {
                "steps": [f"Load {fname}", "Compute df['amount'].mean() directly"],
                "assumed_claims": [f"ASSUMED_DTYPE: {fname} column 'amount' is already float64 with no currency symbols or text NAs."],
            },
        })

    # C3: Dirty category casing & whitespace (8 tasks)
    for i in range(8):
        tid = f"C3_cat_{i:02d}"
        tdir = OUT_DIR / tid
        tdir.mkdir(exist_ok=True)
        variants = ["Bio", " bio ", "BIO", " Bio", "CHEM", "chem ", "Chem"]
        rows = [{"dept": rng.choice(variants), "spend": float(rng.randint(10, 90))} for _ in range(12)]
        rows[0]["dept"] = " bio "
        rows[1]["dept"] = "BIO"
        df = pd.DataFrame(rows)
        fname = f"expenses_{i}.csv"
        df.to_csv(tdir / fname, index=False)
        ans = float(df[df["dept"].str.strip().str.lower() == "bio"]["spend"].sum())
        tasks.append({
            "task_id": tid,
            "category": "C3_dirty_category",
            "is_corrupted_plan": True,
            "work_dir": str(tdir),
            "files": [fname],
            "question": f"Compute the total spend for dept 'Bio' (case-insensitive, ignoring leading/trailing spaces) in {fname}. Print only the final float.",
            "expected_answer": ans,
            "tolerance": 1e-4,
            "upstream_plan": {
                "steps": [f"Load {fname}", "Filter df[df['dept'] == 'Bio']", "Sum 'spend'"],
                "assumed_claims": [f"ASSUMED_VALUES: {fname}.dept contains exact normalized title-case strings without whitespace."],
            },
        })

    # C4: Mixed date formats + invalid tokens (8 tasks)
    for i in range(8):
        tid = f"C4_date_{i:02d}"
        tdir = OUT_DIR / tid
        tdir.mkdir(exist_ok=True)
        q1_dates = ["2025-01-12", "2025/02/18", "2025-03-05", "2025/01/29", "2025-03-28"]
        other_dates = ["2025-04-10", "2025/05/01", "invalid", "TBD"]
        chosen_q1 = rng.sample(q1_dates, k=rng.randint(2, 4))
        chosen_other = rng.sample(other_dates, k=3)
        all_dates = chosen_q1 + chosen_other
        rng.shuffle(all_dates)
        df = pd.DataFrame({"id": list(range(len(all_dates))), "dt": all_dates})
        fname = f"logs_{i}.csv"
        df.to_csv(tdir / fname, index=False)
        ans = float(len(chosen_q1))
        tasks.append({
            "task_id": tid,
            "category": "C4_mixed_dates",
            "is_corrupted_plan": True,
            "work_dir": str(tdir),
            "files": [fname],
            "question": f"Count how many valid rows in {fname} have 'dt' in Q1 2025 (2025-01-01 to 2025-03-31 inclusive). Print only the final number.",
            "expected_answer": ans,
            "tolerance": 1e-4,
            "upstream_plan": {
                "steps": [f"Load {fname}", "Run pd.to_datetime(df['dt']) directly without format='mixed' or errors='coerce'"],
                "assumed_claims": [f"ASSUMED_FORMAT: {fname}.dt strictly follows ISO '%Y-%m-%d' with zero invalid rows."],
            },
        })

    # C5: Two-table join key mismatch (8 tasks)
    for i in range(8):
        tid = f"C5_join_{i:02d}"
        tdir = OUT_DIR / tid
        tdir.mkdir(exist_ok=True)
        u_df = pd.DataFrame({
            "user_id": [101, 102, 103, 104],
            "tier": ["gold", "silver", "gold", "bronze"],
        })
        o_vals = [rng.randint(20, 150) for _ in range(5)]
        o_df = pd.DataFrame({
            "order_id": list(range(5)),
            "user_id": ["00101", "00102", "00103", "00101", "00104"],
            "total": [float(v) for v in o_vals],
        })
        u_name, o_name = f"users_{i}.csv", f"orders_{i}.csv"
        u_df.to_csv(tdir / u_name, index=False)
        o_df.to_csv(tdir / o_name, index=False)
        ans = float(o_vals[0] + o_vals[2] + o_vals[3])
        tasks.append({
            "task_id": tid,
            "category": "C5_join_key_mismatch",
            "is_corrupted_plan": True,
            "work_dir": str(tdir),
            "files": [u_name, o_name],
            "question": f"Join {u_name} and {o_name} on user_id (matching numeric IDs like 101 and '00101') and compute the total order amount for 'gold' tier users. Print only the final float.",
            "expected_answer": ans,
            "tolerance": 1e-4,
            "upstream_plan": {
                "steps": [f"Load {u_name} and {o_name}", "Run pd.merge(users, orders, on='user_id') directly", "Filter tier == 'gold' and sum total"],
                "assumed_claims": [f"ASSUMED_JOIN_KEY: user_id has identical int64 dtype in both {u_name} and {o_name}."],
            },
        })

    manifest = OUT_DIR / "benchmark_v1_manifest.json"
    manifest.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {len(tasks)} Benchmark v1 tasks -> {manifest}")
    return tasks


if __name__ == "__main__":
    generate_benchmark_v1()
