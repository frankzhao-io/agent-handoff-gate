"""Generate publication-grade Figure 1 for Bidirectional Epistemic Handoff Gating.

Produces figures/fig1_handoff_ablation_pareto.pdf and .png:
- Panel a: Pass@1 (%) and False Consensus Rate (%) across A1-A4 (Qwen2.5-Coder-7B) + A2/A4 (Gemini Flash-Lite)
- Panel b: Pass@1 (%) vs. Avg Tokens/Task Pareto Front
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOC_JSON = ROOT / "data" / "benchmark_v1_4arm_results.json"
GEM_JSON = ROOT / "data" / "benchmark_v1_gemini_results.json"
FIG_DIR = ROOT / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    loc = json.loads(LOC_JSON.read_text(encoding="utf-8"))["arms"]
    gem = json.loads(GEM_JSON.read_text(encoding="utf-8"))["arms"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.1), dpi=300, gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Panel a: Pass@1 & FCR
    entries = [
        ("7B\nA1 Reflex.", loc["A1_Single_Reflexion"]),
        ("7B\nA2 Unconst.", loc["A2_Unconstrained_3Agent"]),
        ("7B\nA3 Forward", loc["A3_Forward_Only_Gate"]),
        ("7B\nA4 Bidir.", loc["A4_Bidirectional_Gate"]),
        ("Gemini\nA2 Unconst.", gem["A2_Unconstrained_3Agent"]),
        ("Gemini\nA4 Bidir.", gem["A4_Bidirectional_Gate"]),
    ]
    labels = [e[0] for e in entries]
    pass_pct = [e[1]["pass_at_1"] * 100 for e in entries]
    fcr_pct = [e[1]["false_consensus_rate"] * 100 for e in entries]

    x = np.arange(len(entries))
    w = 0.34
    b1 = ax1.bar(x - w / 2, pass_pct, width=w, label="Pass@1 (%)", color="#2a9d8f")
    b2 = ax1.bar(x + w / 2, fcr_pct, width=w, label="False Consensus Rate (%)", color="#e76f51")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}", ha="center", va="bottom", fontsize=6.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylim(0, 118)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("a  Task Success & False Consensus (N=48)", loc="left", fontweight="bold", fontsize=9)
    ax1.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)

    # Panel b: Pass@1 vs Avg Tokens Pareto Front
    points = [
        ("7B A1", loc["A1_Single_Reflexion"], "#8d99ae", "o", (-14, -12)),
        ("7B A2", loc["A2_Unconstrained_3Agent"], "#e9c46a", "s", (6, -8)),
        ("7B A3", loc["A3_Forward_Only_Gate"], "#f4a261", "^", (-16, 7)),
        ("7B A4 (Ours)", loc["A4_Bidirectional_Gate"], "#264653", "D", (-54, -12)),
        ("Gemini A2", gem["A2_Unconstrained_3Agent"], "#e76f51", "s", (6, 4)),
        ("Gemini A4 (Ours)", gem["A4_Bidirectional_Gate"], "#2a9d8f", "D", (-82, -2)),
    ]
    for name, m, color, marker, offset in points:
        tx = m["avg_tokens_per_task"]
        ty = m["pass_at_1"] * 100
        ax2.scatter(tx, ty, color=color, marker=marker, s=55, zorder=4)
        ax2.annotate(name, (tx, ty), xytext=offset, textcoords="offset points", fontsize=6.8, fontweight="bold" if "Ours" in name else "normal")

    # Pareto frontier line (7B A2 -> 7B A4 -> Gemini A4)
    pareto_x = [loc["A2_Unconstrained_3Agent"]["avg_tokens_per_task"], loc["A4_Bidirectional_Gate"]["avg_tokens_per_task"], gem["A4_Bidirectional_Gate"]["avg_tokens_per_task"]]
    pareto_y = [loc["A2_Unconstrained_3Agent"]["pass_at_1"] * 100, loc["A4_Bidirectional_Gate"]["pass_at_1"] * 100, gem["A4_Bidirectional_Gate"]["pass_at_1"] * 100]
    ax2.plot(pareto_x, pareto_y, linestyle="--", color="#2a9d8f", alpha=0.6, linewidth=1.2, zorder=2)

    ax2.set_xlim(480, 810)
    ax2.set_ylim(25, 110)
    ax2.set_xlabel("Avg. Tokens / Task")
    ax2.set_ylabel("Pass@1 (%)")
    ax2.set_title("b  Accuracy vs. Token Cost Pareto", loc="left", fontweight="bold", fontsize=9)

    fig.tight_layout(w_pad=2.0)
    pdf_out = FIG_DIR / "fig1_handoff_ablation_pareto.pdf"
    png_out = FIG_DIR / "fig1_handoff_ablation_pareto.png"
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_out} and {png_out}")


if __name__ == "__main__":
    main()
