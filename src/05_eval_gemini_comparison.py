"""Exp-03: Compare Local 7B (qwen2.5-coder:7b) vs. Gemini Flash-Lite on Benchmark v1 (N=48).

Evaluates A2 (Unconstrained 3-Agent) and A4 (Bidirectional Epistemic Gate) on
gemini-3.5-flash-lite with JSONL caching and rate-limit throttling, recording
exact token counts, Pass@1, False Consensus Rate (FCR), and Error Propagation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "benchmark_v1" / "benchmark_v1_manifest.json"
CACHE_PATH = ROOT / "data" / "benchmark_v1_cache_gemini.jsonl"
OUT_JSON = ROOT / "data" / "benchmark_v1_gemini_results.json"
MODEL = "gemini-3.5-flash-lite"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


mvp2 = _load("mvp2", "src/02_run_mvp_4arm_ablation.py")
eval4 = _load("eval4", "src/04_eval_4arm_benchmark_v1.py")


def load_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    for env_path in [ROOT / ".env", ROOT.parent / "track3-ai4science" / ".env"]:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def make_gemini_caller(api_key: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

    def call_gemini(system_prompt: str, user_prompt: str, json_mode: bool = False) -> tuple[str, int, float]:
        gen_cfg: dict[str, Any] = {"temperature": 0.0}
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": gen_cfg,
        }
        t0 = time.perf_counter()
        for attempt in range(6):
            try:
                resp = requests.post(
                    url,
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json=payload,
                    timeout=60,
                )
            except requests.exceptions.RequestException:
                time.sleep(10 * (attempt + 1))
                continue
            if resp.status_code in (429, 503):
                time.sleep(20 * (attempt + 1))
                continue
            if not resp.ok:
                raise RuntimeError(f"Gemini API {resp.status_code}: {resp.text[:250]}")
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            tokens = int(data.get("usageMetadata", {}).get("totalTokenCount") or 0)
            time.sleep(2.0)
            return text, tokens, time.perf_counter() - t0
        raise RuntimeError("Exceeded max retries for Gemini API.")

    return call_gemini


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
    api_key = load_api_key()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not found in env or .env.")

    # Inject Gemini caller into both mvp2 and eval4.mvp2
    gemini_caller = make_gemini_caller(api_key)
    mvp2.call_llm = gemini_caller
    eval4.mvp2.call_llm = gemini_caller

    tasks = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cache = load_cache()
    arms = [
        ("A2_Unconstrained_3Agent", lambda t: mvp2.run_a2_unconstrained_3agent(eval4.adapt_task(t))),
        ("A4_Bidirectional_Gate", eval4.run_a4_v2),
    ]

    summary: dict[str, Any] = {"model": MODEL, "n_tasks": len(tasks), "arms": {}}
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
                print(f"[{MODEL} {arm_name} {idx}/{len(tasks)}] {t['task_id']}: passed={res['passed']}, FCR={res['false_consensus']}, tok={res['tokens']}")
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
    print(f"\n=== Gemini ({MODEL}) Benchmark v1 Summary (N={len(tasks)}) ===")
    for arm_name, m in summary["arms"].items():
        print(
            f"{arm_name:24s} | Pass@1={m['pass_at_1']:.4f} | FCR={m['false_consensus_rate']:.4f} | "
            f"ErrProp={m['error_propagation_rate']:.4f} | AvgTok={m['avg_tokens_per_task']:.1f}"
        )


if __name__ == "__main__":
    main()
