"""Unit tests for agent-handoff-gate deterministic epistemic verification & sandbox."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


mvp2 = _load("mvp2", "src/02_run_mvp_4arm_ablation.py")
eval4 = _load("eval4", "src/04_eval_4arm_benchmark_v1.py")


class TestHandoffGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        manifest = ROOT / "data" / "benchmark_v1" / "benchmark_v1_manifest.json"
        cls.tasks = json.loads(manifest.read_text(encoding="utf-8"))

    def test_clean_control_has_no_conflicts(self) -> None:
        c0 = next(t for t in self.tasks if t["category"] == "C0_clean_control")
        conflicts = eval4.check_assumptions_v2(c0["upstream_plan"]["assumed_claims"], c0)
        self.assertEqual(conflicts, [])

    def test_all_fault_categories_trigger_blame_planner(self) -> None:
        for cat in [
            "C1_whitespace_col",
            "C2_currency_na",
            "C3_dirty_category",
            "C4_mixed_dates",
            "C5_join_key_mismatch",
        ]:
            t = next(x for x in self.tasks if x["category"] == cat)
            conflicts = eval4.check_assumptions_v2(t["upstream_plan"]["assumed_claims"], t)
            self.assertTrue(
                any("BLAME_PLANNER:" in c for c in conflicts),
                f"Expected BLAME_PLANNER conflict for {cat}, got {conflicts}",
            )

    def test_verified_facts_exposes_exact_spaced_column(self) -> None:
        c1 = next(t for t in self.tasks if t["category"] == "C1_whitespace_col")
        facts = eval4.inspect_verified_facts_v2(c1)
        self.assertTrue(any("exact_columns=" in f for f in facts))

    def test_sandbox_execution_and_tolerance(self) -> None:
        c0 = next(t for t in self.tasks if t["category"] == "C0_clean_control")
        ans = c0["expected_answer"]
        ok, _, val = mvp2.run_sandbox(f"print({ans})", Path(c0["work_dir"]))
        self.assertTrue(ok)
        self.assertTrue(mvp2.is_correct(val, c0))

        ok_bad, _, _ = mvp2.run_sandbox("raise RuntimeError('boom')", Path(c0["work_dir"]))
        self.assertFalse(ok_bad)


if __name__ == "__main__":
    unittest.main()
