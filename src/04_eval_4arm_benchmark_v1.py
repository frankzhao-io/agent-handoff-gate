"""Benchmark v1 (N=48) 4-Arm Ablation Evaluator (qwen2.5-coder:7b via Ollama).

Reuses sandbox/LLM helpers from 02_run_mvp_4arm_ablation, enhances epistemic
object-column profiling (unique samples + date-format verification) so C4 mixed
dates are caught cleanly, caches every arm/task run to JSONL, and reports
overall + per-category Pass@1, FCR, Error Propagation Rate, and Token Cost.
"""

from __future__ import annotations

import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "benchmark_v1" / "benchmark_v1_manifest.json"
CACHE_PATH = ROOT / "data" / "benchmark_v1_cache_qwen2.5-coder-7b.jsonl"
OUT_JSON = ROOT / "data" / "benchmark_v1_4arm_results.json"

spec2 = importlib.util.spec_from_file_location("mvp2", ROOT / "src" / "02_run_mvp_4arm_ablation.py")
mvp2 = importlib.util.module_from_spec(spec2)
assert spec2 and spec2.loader
spec2.loader.exec_module(mvp2)


def inspect_verified_facts_v2(task: dict[str, Any]) -> list[str]:
    """Observe exact CSV schema, dtypes, and up to 8 distinct values per object column."""
    work_dir = Path(task["work_dir"])
    facts = []
    for fname in task["files"]:
        df = pd.read_csv(work_dir / fname)
        cols = [repr(c) for c in df.columns.tolist()]
        dtypes = {c: str(dt) for c, dt in df.dtypes.items()}
        obj_samples = {
            c: df[c].dropna().astype(str).unique().tolist()[:8]
            for c in df.select_dtypes(include="object").columns
        }
        facts.append(
            f"VERIFIED_FILE({fname}): rows={len(df)}, exact_columns=[{', '.join(cols)}], "
            f"dtypes={json.dumps(dtypes)}, object_unique_samples={json.dumps(obj_samples)}"
        )
    return facts


def check_assumptions_v2(assumed_claims: list[str], task: dict[str, Any]) -> list[str]:
    """Detect contradictions between upstream ASSUMED_* claims and sandbox-observed CSVs."""
    work_dir = Path(task["work_dir"])
    conflicts = []
    for claim in assumed_claims:
        low = claim.lower()
        for fname in task["files"]:
            df = pd.read_csv(work_dir / fname)
            if "assumed_schema" in low:
                spaced = [c for c in df.columns if c != c.strip()]
                if spaced:
                    conflicts.append(
                        f"BLAME_PLANNER: {claim} contradicts VERIFIED exact columns {df.columns.tolist()} in {fname}."
                    )
            if "assumed_dtype" in low:
                obj_cols = [c for c, dt in df.dtypes.items() if dt == object]
                if obj_cols:
                    conflicts.append(
                        f"BLAME_PLANNER: {claim} contradicts VERIFIED object/string dtypes on {obj_cols} "
                        f"(samples={df[obj_cols[0]].head(4).tolist()}) in {fname}."
                    )
            if "assumed_values" in low:
                for c in df.select_dtypes(include="object").columns:
                    vals = df[c].dropna().astype(str).tolist()
                    if any(v != v.strip() or v.lower() != v for v in vals):
                        conflicts.append(
                            f"BLAME_PLANNER: {claim} contradicts VERIFIED unnormalized strings {vals[:5]} in {fname}.{c}."
                        )
            if "assumed_format" in low and "dt" in df.columns:
                try:
                    pd.to_datetime(df["dt"], format="%Y-%m-%d", errors="raise")
                except Exception as exc:
                    conflicts.append(
                        f"BLAME_PLANNER: {claim} failed strict ISO '%Y-%m-%d' check ({type(exc).__name__}: {str(exc)[:80]}). "
                        "Use pd.to_datetime(..., format='mixed', errors='coerce')."
                    )
            if "assumed_join_key" in low and "user_id" in df.columns:
                raw_df = pd.read_csv(work_dir / fname, dtype=str)
                raw_vals = raw_df["user_id"].dropna().astype(str).tolist()
                if df["user_id"].dtype == object or any(v.startswith("0") and len(v) > 1 for v in raw_vals):
                    conflicts.append(
                        f"BLAME_PLANNER: {claim} contradicts VERIFIED raw key format {raw_vals[:3]} in {fname}. "
                        "Normalize join keys (e.g., cast to int) before merging."
                    )
    return conflicts


def adapt_task(t: dict[str, Any]) -> dict[str, Any]:
    """Adapt Benchmark v1 task dict to mvp2's 'corrupted_plan' key."""
    out = dict(t)
    out["corrupted_plan"] = t["upstream_plan"]
    return out


def run_a4_v2(task: dict[str, Any]) -> dict[str, Any]:
    tokens = 0
    t0 = time.perf_counter()
    verified_facts = inspect_verified_facts_v2(task)
    conflicts = check_assumptions_v2(task["upstream_plan"]["assumed_claims"], task)

    if conflicts:
        replan_prompt = (
            f"Task: {task['question']}\n"
            "VERIFIED_FACTS (ground truth from sandbox):\n" + "\n".join(verified_facts) + "\n\n"
            "BACKWARD BLAME COUNTEREXAMPLES:\n" + "\n".join(conflicts) + "\n\n"
            "Produce a corrected execution plan grounded strictly in VERIFIED_FACTS."
        )
        new_plan, tk_p, _ = mvp2.call_llm(mvp2.PLANNER_SYS, replan_prompt)
        tokens += tk_p
    else:
        new_plan = json.dumps(task["upstream_plan"], ensure_ascii=False)

    coder_prompt = (
        f"Task: {task['question']}\n"
        "VERIFIED_FACTS:\n" + "\n".join(verified_facts) + "\n\n"
        f"Verified Plan:\n{new_plan}"
    )
    passed = False
    for _ in range(2):
        raw_code, tk_c, _ = mvp2.call_llm(mvp2.CODER_SYS, coder_prompt)
        tokens += tk_c
        code = mvp2.extract_code(raw_code)
        ok, msg, val = mvp2.run_sandbox(code, Path(task["work_dir"]))
        if ok and mvp2.is_correct(val, task):
            passed = True
            break
        coder_prompt += f"\n\nExecution feedback: {msg}. Ensure exact column names, dtypes, and calculation."

    return {"passed": passed, "false_consensus": False, "tokens": tokens, "sec": round(time.perf_counter() - t0, 2)}


def load_cache() -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[f"{rec['arm']}::{rec['task_id']}"] = rec
    return cache


def append_cache(rec: dict[str, Any]) -> None:
    with CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    tasks = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cache = load_cache()
    arms = [
        ("A1_Single_Reflexion", lambda t: mvp2.run_a1_single_reflexion(adapt_task(t))),
        ("A2_Unconstrained_3Agent", lambda t: mvp2.run_a2_unconstrained_3agent(adapt_task(t))),
        ("A3_Forward_Only_Gate", lambda t: mvp2.run_a3_forward_only_gate(adapt_task(t))),
        ("A4_Bidirectional_Gate", run_a4_v2),
    ]

    summary: dict[str, Any] = {"model": mvp2.MODEL, "n_tasks": len(tasks), "arms": {}}
    for arm_name, fn in arms:
        rows = []
        by_cat: dict[str, list[bool]] = defaultdict(list)
        for idx, t in enumerate(tasks, 1):
            ck = f"{arm_name}::{t['task_id']}"
            if ck not in cache:
                res = fn(t)
                rec = {"arm": arm_name, "task_id": t["task_id"], "category": t["category"], **res}
                append_cache(rec)
                cache[ck] = rec
                print(f"[{arm_name} {idx}/{len(tasks)}] {t['task_id']}: passed={res['passed']}, FCR={res['false_consensus']}, tok={res['tokens']} ({res['sec']}s)")
            r = cache[ck]
            rows.append(r)
            by_cat[t["category"]].append(bool(r["passed"]))

        n = len(rows)
        pass_cnt = sum(1 for r in rows if r["passed"])
        fail_cnt = n - pass_cnt
        fcr_cnt = sum(1 for r in rows if r["false_consensus"])
        tot_tok = sum(r["tokens"] for r in rows)
        tot_sec = round(sum(r["sec"] for r in rows), 2)
        summary["arms"][arm_name] = {
            "pass_at_1": round(pass_cnt / n, 4),
            "false_consensus_rate": round(fcr_cnt / fail_cnt, 4) if fail_cnt else 0.0,
            "error_propagation_rate": round(fail_cnt / n, 4),
            "total_tokens": tot_tok,
            "avg_tokens_per_task": round(tot_tok / n, 1),
            "total_sec": tot_sec,
            "by_category": {
                k: f"{sum(v)}/{len(v)} ({sum(v)/len(v)*100:.1f}%)"
                for k, v in sorted(by_cat.items())
            },
        }

    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== Benchmark v1 (N={len(tasks)}) 4-Arm Summary ({mvp2.MODEL}) ===")
    for arm_name, m in summary["arms"].items():
        print(
            f"{arm_name:24s} | Pass@1={m['pass_at_1']:.4f} | FCR={m['false_consensus_rate']:.4f} | "
            f"ErrProp={m['error_propagation_rate']:.4f} | AvgTok={m['avg_tokens_per_task']:.1f} | Time={m['total_sec']}s"
        )


if __name__ == "__main__":
    main()
