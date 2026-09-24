"""Summarize the lm-evaluation-harness outputs in eval_results/ into the
Base-vs-SFT comparison reported in the paper.

Usage:  python scripts/summarize_results.py
Writes: results/summary.csv and results/summary.md
"""
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (task, metric key, display name)
MAIN = [
    ("arc_easy", "acc_norm,none", "ARC-Easy"),
    ("arc_challenge", "acc_norm,none", "ARC-Challenge"),
    ("hellaswag", "acc_norm,none", "HellaSwag"),
    ("mmlu", "acc,none", "MMLU"),
    ("winogrande", "acc,none", "WinoGrande"),
    ("gsm8k", "exact_match,flexible-extract", "GSM8K"),
]
MMLU_CATS = [
    ("mmlu_humanities", "Humanities"),
    ("mmlu_other", "Other"),
    ("mmlu_social_sciences", "Social Sciences"),
    ("mmlu_stem", "STEM"),
]


def load(condition: str) -> dict:
    files = sorted(glob.glob(str(ROOT / "eval_results" / condition / "*" / "results_*.json")))
    if not files:
        raise FileNotFoundError(f"No results file found for '{condition}'")
    return json.load(open(files[-1]))["results"]


def stderr_key(metric: str) -> str:
    name, flt = metric.split(",")
    return f"{name}_stderr,{flt}"


def main() -> None:
    base, sft = load("base"), load("sft")
    rows = []
    for task, metric, name in MAIN:
        b, s = base[task][metric] * 100, sft[task][metric] * 100
        rows.append({
            "benchmark": name, "metric": metric.split(",")[0],
            "base": round(b, 2), "base_stderr": round(base[task][stderr_key(metric)] * 100, 2),
            "sft": round(s, 2), "sft_stderr": round(sft[task][stderr_key(metric)] * 100, 2),
            "delta_pp": round(round(s, 2) - round(b, 2), 2),  # difference of reported (rounded) scores
        })
    for task, name in MMLU_CATS:
        b, s = base[task]["acc,none"] * 100, sft[task]["acc,none"] * 100
        rows.append({
            "benchmark": f"MMLU – {name}", "metric": "acc",
            "base": round(b, 2), "base_stderr": round(base[task]["acc_stderr,none"] * 100, 2),
            "sft": round(s, 2), "sft_stderr": round(sft[task]["acc_stderr,none"] * 100, 2),
            "delta_pp": round(round(s, 2) - round(b, 2), 2),  # difference of reported (rounded) scores
        })

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    cols = list(rows[0].keys())
    with open(out / "summary.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    md = ["| Benchmark | Metric | Base (%) | SFT (%) | Δ (pp) |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['benchmark']} | {r['metric']} | {r['base']:.2f} ± {r['base_stderr']:.2f} "
                  f"| {r['sft']:.2f} ± {r['sft_stderr']:.2f} | {r['delta_pp']:+.2f} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
