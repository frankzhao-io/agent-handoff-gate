"""MVP 4-Arm Ablation on Corrupted-Plan Fault-Injection Tasks (qwen2.5-coder:7b via Ollama).

Compares:
- A1: Single-Agent + Reflexion (sandbox error feedback, up to 2 retries)
- A2: Unconstrained 3-Agent (Corrupted Planner -> Coder -> LLM Reviewer, no sandbox gate)
- A3: Forward-Only Gate (ToolGate-style: forward execution check, retries Coder only)
- A4: Bidirectional Epistemic Gate (VERIFIED_* vs ASSUMED_* split + backward blame routing to Planner)

Measures Pass@1, False Consensus Rate (FCR), Error Propagation Rate, and Total Tokens.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "mvp_tasks" / "mvp_manifest.json"
OUT_JSON = ROOT / "data" / "mvp_4arm_results.json"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.environ.get("OLLAMA_CODER_MODEL", "qwen2.5-coder:7b")


def call_llm(system_prompt: str, user_prompt: str, json_mode: bool = False) -> tuple[str, int, float]:
    """Call local Ollama chat API and return (text, total_tokens, elapsed_sec)."""
    payload: dict[str, Any] = {
        "model": MODEL,
        "stream": False,
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if json_mode:
        payload["format"] = "json"
    t0 = time.perf_counter()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    elapsed = time.perf_counter() - t0
    data = resp.json()
    text = data.get("message", {}).get("content", "")
    tokens = int(data.get("prompt_eval_count") or 0) + int(data.get("eval_count") or 0)
    return text, tokens, elapsed


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    return (m.group(1) if m else text).strip()


def run_sandbox(code: str, work_dir: Path) -> tuple[bool, str, float | None]:
    """Execute Python code in work_dir with 10s timeout; parse final float if possible."""
    script = work_dir / "_tmp_solution.py"
    script.write_text(code, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired after 10s", None
    finally:
        script.unlink(missing_ok=True)

    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()[-500:], None

    out = proc.stdout.strip()
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", out)
    if not nums:
        return False, f"No numeric output printed. stdout={out!r}", None
    return True, out, float(nums[-1])


def is_correct(pred: float | None, task: dict[str, Any]) -> bool:
    if pred is None:
        return False
    return abs(pred - float(task["expected_answer"])) <= float(task["tolerance"])


def inspect_verified_facts(task: dict[str, Any]) -> list[str]:
    """Epistemic Gate: deterministically observe CSV schemas, dtypes, and sample values."""
    work_dir = Path(task["work_dir"])
    facts = []
    for fname in task["files"]:
        df = pd.read_csv(work_dir / fname)
        cols = [repr(c) for c in df.columns.tolist()]
        dtypes = {c: str(dt) for c, dt in df.dtypes.items()}
        samples = df.head(3).to_dict(orient="records")
        facts.append(
            f"VERIFIED_FILE({fname}): rows={len(df)}, exact_columns=[{', '.join(cols)}], "
            f"dtypes={json.dumps(dtypes)}, head3={json.dumps(samples, default=str)}"
        )
    return facts


def check_assumptions_against_facts(assumed_claims: list[str], verified_facts: list[str], task: dict[str, Any]) -> list[str]:
    """Deterministic contradiction detector between upstream ASSUMED_* and sandbox VERIFIED_*."""
    work_dir = Path(task["work_dir"])
    conflicts = []
    for claim in assumed_claims:
        low = claim.lower()
        for fname in task["files"]:
            df = pd.read_csv(work_dir / fname)
            if "assumed_schema" in low:
                spaced = [c for c in df.columns if c != c.strip()]
                if spaced:
                    conflicts.append(f"BLAME_PLANNER: {claim} contradicts VERIFIED exact columns {df.columns.tolist()} in {fname}.")
            if "assumed_dtype" in low:
                obj_cols = [c for c, dt in df.dtypes.items() if dt == object]
                if obj_cols:
                    conflicts.append(f"BLAME_PLANNER: {claim} contradicts VERIFIED object/string dtypes on {obj_cols} in {fname}.")
            if "assumed_values" in low:
                for c in df.select_dtypes(include="object").columns:
                    vals = df[c].dropna().astype(str).tolist()
                    if any(v != v.strip() or v.lower() != v for v in vals):
                        conflicts.append(f"BLAME_PLANNER: {claim} contradicts VERIFIED unnormalized strings {vals[:4]} in {fname}.{c}.")
            if "assumed_format" in low:
                conflicts.append(f"BLAME_PLANNER: {claim} unverified without format='mixed'/errors='coerce' check against sample rows in {fname}.")
            if "assumed_join_key" in low:
                if "user_id" in df.columns and df["user_id"].dtype == object:
                    conflicts.append(f"BLAME_PLANNER: {claim} contradicts VERIFIED string dtype on user_id={df['user_id'].head(2).tolist()} in {fname}.")
            if "assumed_uniqueness" in low:
                for c in df.columns:
                    if df[c].duplicated().any():
                        conflicts.append(f"BLAME_PLANNER: {claim} contradicts VERIFIED duplicate values in {fname}.{c}.")
    return conflicts


CODER_SYS = (
    "You are a Python data-analysis Coder. Write standalone Python code using pandas/numpy "
    "that reads the CSV file(s) from the current directory and prints ONLY the final numeric answer. "
    "Output ONLY a ```python ... ``` block."
)

REVIEWER_SYS = (
    "You are a strict code Reviewer. Given the task, upstream plan, and Python code, decide if the code "
    'should be approved. Respond ONLY in JSON: {"approved": true/false, "reason": "..."}'
)

PLANNER_SYS = (
    "You are a Data Analysis Planner. Using ONLY the VERIFIED_* facts provided, produce a concise, "
    "accurate step-by-step plan addressing any BLAME_PLANNER counterexamples."
)


def run_a1_single_reflexion(task: dict[str, Any]) -> dict[str, Any]:
    plan_txt = json.dumps(task["corrupted_plan"], ensure_ascii=False)
    prompt = f"Task: {task['question']}\nFiles: {task['files']}\nInitial Plan (may contain errors): {plan_txt}"
    tokens = 0
    t0 = time.perf_counter()
    passed = False
    for attempt in range(3):
        raw, tk, _ = call_llm(CODER_SYS, prompt)
        tokens += tk
        code = extract_code(raw)
        ok, msg, val = run_sandbox(code, Path(task["work_dir"]))
        if ok and is_correct(val, task):
            passed = True
            break
        prompt = f"Task: {task['question']}\nPrevious code failed or produced wrong/error output:\n{msg}\nFix the code and inspect CSV columns/dtypes in code if needed."
    return {"passed": passed, "false_consensus": False, "tokens": tokens, "sec": round(time.perf_counter() - t0, 2)}


def run_a2_unconstrained_3agent(task: dict[str, Any]) -> dict[str, Any]:
    plan_txt = json.dumps(task["corrupted_plan"], ensure_ascii=False)
    tokens = 0
    t0 = time.perf_counter()
    raw_code, tk1, _ = call_llm(CODER_SYS, f"Task: {task['question']}\nFollow this Planner specification:\n{plan_txt}")
    tokens += tk1
    code = extract_code(raw_code)

    raw_rev, tk2, _ = call_llm(REVIEWER_SYS, f"Task: {task['question']}\nPlan: {plan_txt}\nCode:\n{code}", json_mode=True)
    tokens += tk2
    try:
        approved = bool(json.loads(raw_rev).get("approved", False))
    except Exception:
        approved = False

    ok, _, val = run_sandbox(code, Path(task["work_dir"]))
    passed = ok and is_correct(val, task)
    false_consensus = approved and (not passed)
    return {"passed": passed, "false_consensus": false_consensus, "tokens": tokens, "sec": round(time.perf_counter() - t0, 2)}


def run_a3_forward_only_gate(task: dict[str, Any]) -> dict[str, Any]:
    plan_txt = json.dumps(task["corrupted_plan"], ensure_ascii=False)
    tokens = 0
    t0 = time.perf_counter()
    prompt = f"Task: {task['question']}\nFollow this Planner specification:\n{plan_txt}"
    passed = False
    approved_after_fail = False
    for attempt in range(2):
        raw_code, tk1, _ = call_llm(CODER_SYS, prompt)
        tokens += tk1
        code = extract_code(raw_code)
        ok, msg, val = run_sandbox(code, Path(task["work_dir"]))
        if ok and is_correct(val, task):
            passed = True
            break
        prompt = f"Task: {task['question']}\nPlan: {plan_txt}\nForward runtime error: {msg}\nFix your code."
    if not passed:
        # Check if Reviewer still rubber-stamps silent wrong answers (e.g., T3=0 or T6=wrong mean)
        raw_rev, tk2, _ = call_llm(REVIEWER_SYS, f"Task: {task['question']}\nPlan: {plan_txt}\nCode:\n{code}", json_mode=True)
        tokens += tk2
        try:
            approved_after_fail = bool(json.loads(raw_rev).get("approved", False))
        except Exception:
            approved_after_fail = False
    return {"passed": passed, "false_consensus": approved_after_fail, "tokens": tokens, "sec": round(time.perf_counter() - t0, 2)}


def run_a4_bidirectional_gate(task: dict[str, Any]) -> dict[str, Any]:
    tokens = 0
    t0 = time.perf_counter()
    verified_facts = inspect_verified_facts(task)
    conflicts = check_assumptions_against_facts(task["corrupted_plan"]["assumed_claims"], verified_facts, task)

    if conflicts:
        # Backward Counterexample Routing: invalidate upstream ASSUMED_* and re-invoke Planner
        replan_prompt = (
            f"Task: {task['question']}\n"
            f"VERIFIED_FACTS (ground truth from sandbox):\n" + "\n".join(verified_facts) + "\n\n"
            f"BACKWARD BLAME COUNTEREXAMPLES:\n" + "\n".join(conflicts) + "\n\n"
            "Produce a corrected execution plan grounded strictly in VERIFIED_FACTS."
        )
        new_plan, tk_p, _ = call_llm(PLANNER_SYS, replan_prompt)
        tokens += tk_p
    else:
        new_plan = json.dumps(task["corrupted_plan"], ensure_ascii=False)

    coder_prompt = (
        f"Task: {task['question']}\n"
        f"VERIFIED_FACTS:\n" + "\n".join(verified_facts) + "\n\n"
        f"Verified Plan:\n{new_plan}"
    )
    passed = False
    for attempt in range(2):
        raw_code, tk_c, _ = call_llm(CODER_SYS, coder_prompt)
        tokens += tk_c
        code = extract_code(raw_code)
        ok, msg, val = run_sandbox(code, Path(task["work_dir"]))
        if ok and is_correct(val, task):
            passed = True
            break
        coder_prompt += f"\n\nExecution feedback: {msg}. Ensure exact column names, dtypes, and calculation."

    return {"passed": passed, "false_consensus": False, "tokens": tokens, "sec": round(time.perf_counter() - t0, 2)}


def main() -> None:
    tasks = json.loads(MANIFEST.read_text(encoding="utf-8"))
    arms = [
        ("A1_Single_Reflexion", run_a1_single_reflexion),
        ("A2_Unconstrained_3Agent", run_a2_unconstrained_3agent),
        ("A3_Forward_Only_Gate", run_a3_forward_only_gate),
        ("A4_Bidirectional_Gate", run_a4_bidirectional_gate),
    ]

    summary: dict[str, Any] = {"model": MODEL, "n_tasks": len(tasks), "arms": {}}
    for arm_name, fn in arms:
        rows = []
        for t in tasks:
            res = fn(t)
            rows.append({"task_id": t["task_id"], **res})
            print(f"[{arm_name}] {t['task_id']}: passed={res['passed']}, FCR={res['false_consensus']}, tokens={res['tokens']} ({res['sec']}s)")
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
            "details": rows,
        }

    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== MVP 4-Arm Summary (qwen2.5-coder:7b) ===")
    for arm_name, m in summary["arms"].items():
        print(f"{arm_name:24s} | Pass@1={m['pass_at_1']:.2f} | FCR={m['false_consensus_rate']:.2f} | ErrProp={m['error_propagation_rate']:.2f} | AvgTok={m['avg_tokens_per_task']:.0f} | Time={m['total_sec']}s")


if __name__ == "__main__":
    main()
