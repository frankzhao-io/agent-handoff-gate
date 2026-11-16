"""MVP Multi-Step Data-Analysis Benchmark with Hidden Schema Traps & Corrupted-Plan Fault Injection.

Generates 6 realistic tabular analysis tasks (T1-T6) in data/mvp_tasks/.
Each task includes:
- Real CSV file(s) with hidden schema/dtype/whitespace/key traps
- Ground-truth numerical answer computed deterministically
- A plausible-sounding CORRUPTED upstream Planner assumption (ASSUMED_*) to
  directly measure False Consensus Rate (FCR) and Error Propagation Rate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "mvp_tasks"


def build_tasks() -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []

    # T1: Trailing whitespace in column name (" revenue ")
    t1_dir = OUT_DIR / "T1_whitespace_col"
    t1_dir.mkdir(exist_ok=True)
    df1 = pd.DataFrame({
        "region": ["North", "South", "North", "East", "North"],
        " revenue ": [120.0, 85.0, 230.0, 95.0, 150.0],
    })
    df1.to_csv(t1_dir / "sales.csv", index=False)
    tasks.append({
        "task_id": "T1_whitespace_col",
        "work_dir": str(t1_dir),
        "files": ["sales.csv"],
        "question": "Compute the total revenue for region == 'North' in sales.csv. Print only the final float.",
        "expected_answer": 500.0,
        "tolerance": 1e-4,
        "corrupted_plan": {
            "steps": ["Load sales.csv", "Filter rows where region == 'North'", "Sum the 'revenue' column"],
            "assumed_claims": ["ASSUMED_SCHEMA: sales.csv has columns ['region', 'revenue'] with float dtype."],
        },
    })

    # T2: Currency string + 'N/A' in amount column
    t2_dir = OUT_DIR / "T2_currency_na"
    t2_dir.mkdir(exist_ok=True)
    df2 = pd.DataFrame({
        "tx_id": [1, 2, 3, 4, 5],
        "amount": ["$1,200.00", "$800.00", "N/A", "$1,000.00", "N/A"],
    })
    df2.to_csv(t2_dir / "transactions.csv", index=False)
    tasks.append({
        "task_id": "T2_currency_na",
        "work_dir": str(t2_dir),
        "files": ["transactions.csv"],
        "question": "Compute the mean of valid numeric values in the 'amount' column of transactions.csv. Print only the final float.",
        "expected_answer": 1000.0,
        "tolerance": 1e-4,
        "corrupted_plan": {
            "steps": ["Load transactions.csv", "Compute df['amount'].mean() directly"],
            "assumed_claims": ["ASSUMED_DTYPE: transactions.csv column 'amount' is already float64 with no currency symbols."],
        },
    })

    # T3: Whitespace + case mismatch in category key
    t3_dir = OUT_DIR / "T3_dirty_category"
    t3_dir.mkdir(exist_ok=True)
    df3 = pd.DataFrame({
        "category": ["Bio", " bio ", "CHEM", "BIO", "chem ", "Bio"],
        "cost": [10.0, 20.0, 50.0, 30.0, 25.0, 40.0],
    })
    df3.to_csv(t3_dir / "lab_orders.csv", index=False)
    tasks.append({
        "task_id": "T3_dirty_category",
        "work_dir": str(t3_dir),
        "files": ["lab_orders.csv"],
        "question": "Compute the total cost for all rows belonging to category 'Bio' (case-insensitive, ignoring leading/trailing spaces) in lab_orders.csv. Print only the final float.",
        "expected_answer": 100.0,
        "tolerance": 1e-4,
        "corrupted_plan": {
            "steps": ["Load lab_orders.csv", "Filter df[df['category'] == 'Bio']", "Sum cost"],
            "assumed_claims": ["ASSUMED_VALUES: category column is already normalized to exact title-case strings without whitespace."],
        },
    })

    # T4: Mixed date formats + invalid string
    t4_dir = OUT_DIR / "T4_mixed_dates"
    t4_dir.mkdir(exist_ok=True)
    df4 = pd.DataFrame({
        "event_id": [1, 2, 3, 4, 5, 6],
        "event_date": ["2025-01-10", "2025/02/15", "invalid", "2025-03-20", "2025-04-05", "2025-02-28"],
    })
    df4.to_csv(t4_dir / "events.csv", index=False)
    tasks.append({
        "task_id": "T4_mixed_dates",
        "work_dir": str(t4_dir),
        "files": ["events.csv"],
        "question": "Count how many valid events occurred in Q1 2025 (2025-01-01 to 2025-03-31 inclusive) in events.csv. Print only the final integer/float.",
        "expected_answer": 4.0,
        "tolerance": 1e-4,
        "corrupted_plan": {
            "steps": ["Load events.csv", "Parse event_date with pd.to_datetime(df['event_date']) without format='mixed' or errors='coerce'"],
            "assumed_claims": ["ASSUMED_FORMAT: event_date strictly follows ISO '%Y-%m-%d' with zero invalid rows."],
        },
    })

    # T5: Two-table join key dtype mismatch (int vs zero-padded string)
    t5_dir = OUT_DIR / "T5_join_key_mismatch"
    t5_dir.mkdir(exist_ok=True)
    pd.DataFrame({"user_id": [101, 102, 103], "tier": ["gold", "silver", "gold"]}).to_csv(t5_dir / "users.csv", index=False)
    pd.DataFrame({"order_id": [1, 2, 3, 4], "user_id": ["00101", "00102", "00103", "00101"], "total": [40.0, 25.0, 60.0, 100.0]}).to_csv(t5_dir / "orders.csv", index=False)
    tasks.append({
        "task_id": "T5_join_key_mismatch",
        "work_dir": str(t5_dir),
        "files": ["users.csv", "orders.csv"],
        "question": "Join users.csv and orders.csv on user_id (matching numeric IDs like 101 and '00101') and compute the total order amount for 'gold' tier users. Print only the final float.",
        "expected_answer": 200.0,
        "tolerance": 1e-4,
        "corrupted_plan": {
            "steps": ["Load users.csv and orders.csv", "Directly run pd.merge(users, orders, on='user_id')", "Filter tier == 'gold' and sum total"],
            "assumed_claims": ["ASSUMED_JOIN_KEY: user_id has identical int64 dtype in both users.csv and orders.csv."],
        },
    })

    # T6: Duplicate sample_id + NaN score
    t6_dir = OUT_DIR / "T6_dedup_median"
    t6_dir.mkdir(exist_ok=True)
    df6 = pd.DataFrame({
        "sample_id": ["S1", "S1", "S2", "S3", "S4", "S4"],
        "score": [10.0, 10.0, 20.0, 50.0, float("nan"), float("nan")],
    })
    df6.to_csv(t6_dir / "assay.csv", index=False)
    tasks.append({
        "task_id": "T6_dedup_median",
        "work_dir": str(t6_dir),
        "files": ["assay.csv"],
        "question": "Deduplicate assay.csv by sample_id (keeping first), drop missing scores, and compute the median score. Print only the final float.",
        "expected_answer": 20.0,
        "tolerance": 1e-4,
        "corrupted_plan": {
            "steps": ["Load assay.csv", "Compute df['score'].mean() without deduplicating sample_id"],
            "assumed_claims": ["ASSUMED_UNIQUENESS: sample_id is already unique and metric requested is mean."],
        },
    })

    manifest = OUT_DIR / "mvp_manifest.json"
    manifest.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {len(tasks)} MVP fault-injection tasks -> {manifest}")
    return tasks


if __name__ == "__main__":
    build_tasks()
