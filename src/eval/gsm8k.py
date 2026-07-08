"""Score GSM8K outputs LOCALLY (Mac, no GPU). Reads a run_all jsonl (records carry
`output`, `reference`, `config_key`), extracts the predicted number, compares to gold,
and reports accuracy per config_key. Also dumps a scores jsonl for plot.py.

    python src/eval/gsm8k.py --in results/e2.jsonl --out results/e2_scores.jsonl
"""
import argparse
import json
import re
from collections import defaultdict


def extract_pred(text):
    """Take the number after the first '####' if present, else the last number seen."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        val = m.group(1)
    else:
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
        if not nums:
            return None
        val = nums[-1]
    return val.replace(",", "").rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    correct = defaultdict(int)
    total = defaultdict(int)
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "reference" not in r:
                continue
            ck = r.get("config_key", "default")
            pred = extract_pred(r["output"])
            gold = str(r["reference"]).replace(",", "").strip()
            total[ck] += 1
            if pred is not None and pred == gold:
                correct[ck] += 1

    rows = []
    print(f"{'config_key':<40}{'acc':>8}{'n':>6}")
    for ck in sorted(total):
        acc = correct[ck] / total[ck]
        print(f"{ck:<40}{100*acc:>7.1f}%{total[ck]:>6}")
        rows.append({"config_key": ck, "accuracy": acc, "n": total[ck],
                     "correct": correct[ck], "metric": "gsm8k"})

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"\nscores -> {args.out}")


if __name__ == "__main__":
    main()
