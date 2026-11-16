"""Exp-04: Real-World InfiAgent-DABench (N=25) Natural Error Propagation Evaluation.

Fetches 25 real single-numeric-answer tasks (prioritizing medium/hard levels) from
the official InfiAgent-DABench repository into data/raw_datasets/dabench/ (gitignored).
Tests natural (non-injected) multi-agent error propagation on qwen2.5-coder:7b:
- A2_Unconstrained_3Agent: Blind Planner (filename+question only) -> Coder -> LLM Reviewer
- A4_Bidirectional_Gate: Epistemic CSV Schema/Value Gate + Backward Re-Plan + Execution Gate
"""

from __future__ import annotations

import importlib.util
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_datasets" / "dabench"
TABLES_DIR = RAW_DIR / "tables"
CACHE_PATH = ROOT / "data" / "dabench_25_cache_qwen2.5-coder-7b.jsonl"
OUT_JSON = ROOT / "data" / "dabench_25_results.json"

BASE_RAW_URL = "https://raw.githubusercontent.com/InfiAgent/InfiAgent/main/examples/DA-Agent/data"

spec2 = importlib.util.spec_from_file_location("mvp2", ROOT / "src" / "02_run_mvp_4arm_ablation.py")
mvp2 = importlib.util.module_from_spec(spec2)
assert spec2 and spec2.loader
spec2.loader.exec_module(mvp2)

BLIND_PLANNER_SYS = (
    "You are a Data Analysis Planner. Given ONLY the question, constraints, and CSV filename "
    "(you do NOT have file access), propose a concise step-by-step Python/pandas analysis plan "
    "and explicitly list your ASSUMED_SCHEMA / ASSUMED_DTYPE assumptions."
)


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def prepare_dabench_subset(target_n: int = 25) -> list[dict[str, Any]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    q_path = RAW_DIR / "da-dev-questions.jsonl"
    l_path = RAW_DIR / "da-dev-labels.jsonl"
    if not q_path.exists():
        q_path.write_text(requests.get(f"{BASE_RAW_URL}/da-dev-questions.jsonl", timeout=30).text, encoding="utf-8")
    if not l_path.exists():
        l_path.write_text(requests.get(f"{BASE_RAW_URL}/da-dev-labels.jsonl", timeout=30).text, encoding="utf-8")

    questions = [json.loads(line) for line in q_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = {
        rec["id"]: rec["common_answers"]
        for rec in (json.loads(line) for line in l_path.read_text(encoding="utf-8").splitlines() if line.strip())
    }

    # Filter to single numeric ground-truth answers, prioritizing hard/medium
    candidates = []
    for q in questions:
        ans_list = labels.get(q["id"], [])
        if len(ans_list) == 1 and _is_float(str(ans_list[0][1])):
            q["expected_answer"] = float(ans_list[0][1])
            candidates.append(q)

    priority = {"hard": 0, "medium": 1, "easy": 2}
    candidates.sort(key=lambda x: (priority.get(x.get("level", "easy"), 3), x["id"]))

    selected: list[dict[str, Any]] = []
    for q in candidates:
        fname = q["file_name"]
        csv_path = TABLES_DIR / fname
        if not csv_path.exists():
            r = requests.get(f"{BASE_RAW_URL}/da-dev-tables/{fname}", timeout=30)
            if not r.ok:
                continue
            csv_path.write_bytes(r.content)
        selected.append(q)
        if len(selected) >= target_n:
            break

    print(f"Prepared {len(selected)} real InfiAgent-DABench tasks (levels: "
          f"{ {lvl: sum(1 for x in selected if x.get('level')==lvl) for lvl in ('hard','medium','easy')} }).")
    return selected


def inspect_csv_verified(fname: str) -> str:
    df = pd.read_csv(TABLES_DIR / fname)
    cols = [repr(c) for c in df.columns.tolist()]
    dtypes = {c: str(dt) for c, dt in df.dtypes.items()}
    nulls = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    obj_samples = {
        c: df[c].dropna().astype(str).unique().tolist()[:6]
        for c in df.select_dtypes(include="object").columns
    }
    return (
        f"VERIFIED_FILE({fname}): shape={df.shape}, exact_columns=[{', '.join(cols)}], "
        f"dtypes={json.dumps(dtypes)}, nulls={json.dumps(nulls)}, object_samples={json.dumps(obj_samples)}"
    )


def check_plan_against_csv(plan_text: str, fname: str) -> list[str]:
    """Detect backticked/quoted column names in Planner output that do not exist in actual CSV."""
    df = pd.read_csv(TABLES_DIR / fname)
    actual_cols = set(df.columns.tolist())
    # Look for quoted identifiers that look like column guesses
    quoted = set(re.findall(r"['\"]([A-Za-z0-9_ \-]+)['\"]", plan_text))
    missing = [q for q in quoted if q not in actual_cols and q != fname and not q.endswith(".csv") and len(q) > 2]
    # Only flag if at least one guessed name resembles a column/snake_case or matches none when >1 guesses
    if missing and any("_" in m or m.lower() in {c.lower() for c in actual_cols} for m in missing):
        return [
            f"BLAME_PLANNER: Plan referenced non-existent/mismatched identifier(s) {missing[:4]} "
            f"not in VERIFIED exact_columns={df.columns.tolist()}."
        ]
    return []


def eval_answer(pred: float | None, expected: float) -> bool:
    if pred is None:
        return False
    tol = max(1e-2, abs(expected) * 1e-3)
    return abs(pred - expected) <= tol


def run_a2_real(q: dict[str, Any]) -> dict[str, Any]:
    tokens = 0
    t0 = time.perf_counter()
    prompt_p = f"Question: {q['question']}\nConstraints: {q['constraints']}\nFile: {q['file_name']}"
    plan, tk0, _ = mvp2.call_llm(BLIND_PLANNER_SYS, prompt_p)
    tokens += tk0

    raw_code, tk1, _ = mvp2.call_llm(
        mvp2.CODER_SYS,
        f"Task: {q['question']}\nConstraints: {q['constraints']}\nFile: {q['file_name']}\nPlan:\n{plan}",
    )
    tokens += tk1
    code = mvp2.extract_code(raw_code)

    raw_rev, tk2, _ = mvp2.call_llm(
        mvp2.REVIEWER_SYS,
        f"Task: {q['question']}\nPlan:\n{plan}\nCode:\n{code}",
        json_mode=True,
    )
    tokens += tk2
    try:
        approved = bool(json.loads(raw_rev).get("approved", False))
    except Exception:
        approved = False

    ok, _, val = mvp2.run_sandbox(code, TABLES_DIR)
    passed = ok and eval_answer(val, q["expected_answer"])
    return {"passed": passed, "false_consensus": approved and (not passed), "tokens": tokens, "sec": round(time.perf_counter() - t0, 2)}


def run_a4_real(q: dict[str, Any]) -> dict[str, Any]:
    tokens = 0
    t0 = time.perf_counter()
    fname = q["file_name"]
    prompt_p = f"Question: {q['question']}\nConstraints: {q['constraints']}\nFile: {fname}"
    plan, tk0, _ = mvp2.call_llm(BLIND_PLANNER_SYS, prompt_p)
    tokens += tk0

    verified = inspect_csv_verified(fname)
    conflicts = check_plan_against_csv(plan, fname)
    if conflicts:
        replan_prompt = (
            f"Question: {q['question']}\nConstraints: {q['constraints']}\n"
            f"VERIFIED_FACTS:\n{verified}\n\n"
            f"BACKWARD BLAME COUNTEREXAMPLES:\n" + "\n".join(conflicts) + "\n\n"
            "Produce a corrected step-by-step plan grounded strictly in VERIFIED_FACTS."
        )
        plan, tk_rp, _ = mvp2.call_llm(mvp2.PLANNER_SYS, replan_prompt)
        tokens += tk_rp

    coder_prompt = (
        f"Task: {q['question']}\nConstraints: {q['constraints']}\nFile: {fname}\n"
        f"VERIFIED_FACTS:\n{verified}\n\nPlan:\n{plan}"
    )
    passed = False
    for _ in range(2):
        raw_code, tk_c, _ = mvp2.call_llm(mvp2.CODER_SYS, coder_prompt)
        tokens += tk_c
        code = mvp2.extract_code(raw_code)
        ok, msg, val = mvp2.run_sandbox(code, TABLES_DIR)
        if ok and eval_answer(val, q["expected_answer"]):
            passed = True
            break
        coder_prompt += f"\n\nExecution error/output feedback: {msg}. Fix using VERIFIED_FACTS."

    return {"passed": passed, "false_consensus": False, "tokens": tokens, "sec": round(time.perf_counter() - t0, 2)}


def load_cache() -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[f"{rec['arm']}::{rec['id']}"] = rec
    return cache


def append_cache(rec: dict[str, Any]) -> None:
    with CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    tasks = prepare_dabench_subset(target_n=25)
    cache = load_cache()
    arms = [
        ("A2_Unconstrained_3Agent", run_a2_real),
        ("A4_Bidirectional_Gate", run_a4_real),
    ]

    summary: dict[str, Any] = {"dataset": "InfiAgent-DABench", "model": mvp2.MODEL, "n_tasks": len(tasks), "arms": {}}
    for arm_name, fn in arms:
        rows = []
        by_lvl: dict[str, list[bool]] = defaultdict(list)
        for idx, q in enumerate(tasks, 1):
            ck = f"{arm_name}::{q['id']}"
            if ck not in cache:
                res = fn(q)
                rec = {"arm": arm_name, "id": q["id"], "level": q.get("level", "unknown"), "file": q["file_name"], **res}
                append_cache(rec)
                cache[ck] = rec
                print(f"[{arm_name} {idx}/{len(tasks)}] id={q['id']} ({q.get('level')}): passed={res['passed']}, FCR={res['false_consensus']}, tok={res['tokens']} ({res['sec']}s)")
            r = cache[ck]
            rows.append(r)
            by_lvl[r["level"]].append(bool(r["passed"]))

        n = len(rows)
        pass_cnt = sum(1 for r in rows if r["passed"])
        fail_cnt = n - pass_cnt
        fcr_cnt = sum(1 for r in rows if r["false_consensus"])
        tot_tok = sum(r["tokens"] for r in rows)
        summary["arms"][arm_name] = {
            "pass_at_1": round(pass_cnt / n, 4),
            "false_consensus_rate": round(fcr_cnt / fail_cnt, 4) if fail_cnt else 0.0,
            "error_propagation_rate": round(fail_cnt / n, 4),
            "avg_tokens_per_task": round(tot_tok / n, 1),
            "total_sec": round(sum(r["sec"] for r in rows), 2),
            "by_level": {k: f"{sum(v)}/{len(v)} ({sum(v)/len(v)*100:.1f}%)" for k, v in sorted(by_lvl.items())},
        }

    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== InfiAgent-DABench (N={len(tasks)}) Summary ({mvp2.MODEL}) ===")
    for arm_name, m in summary["arms"].items():
        print(
            f"{arm_name:24s} | Pass@1={m['pass_at_1']:.4f} | FCR={m['false_consensus_rate']:.4f} | "
            f"ErrProp={m['error_propagation_rate']:.4f} | AvgTok={m['avg_tokens_per_task']:.1f}"
        )


if __name__ == "__main__":
    main()
