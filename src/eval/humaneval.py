"""Score HumanEval outputs LOCALLY by executing unit tests. Reads a run_all jsonl where
each record has `output` (the completion), `prompt`, and `reference={test, entry_point}`.
Reports pass@1 per config_key and dumps a scores jsonl.

SECURITY: this EXECUTES model-generated code in a subprocess. Run only in a throwaway
environment (the pod or a container). Each program runs with a timeout.

    python src/eval/humaneval.py --in results/e3.jsonl --out results/e3_scores.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n@"]


def truncate(completion):
    """HumanEval completions should stop at the first out-of-function token."""
    cut = len(completion)
    for s in STOPS:
        i = completion.find(s)
        if i != -1:
            cut = min(cut, i)
    return completion[:cut]


def passes(prompt, completion, test, entry_point, timeout=10):
    program = prompt + truncate(completion) + "\n" + test + f"\ncheck({entry_point})\n"
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as f:
            f.write(program)
            path = f.name
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=int, default=10)
    args = ap.parse_args()

    correct = defaultdict(int)
    total = defaultdict(int)
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ref = r.get("reference")
            if not isinstance(ref, dict):
                continue
            ck = r.get("config_key", "default")
            ok = passes(r["prompt"], r["output"], ref["test"], ref["entry_point"],
                        args.timeout)
            total[ck] += 1
            correct[ck] += int(ok)

    rows = []
    print(f"{'config_key':<40}{'pass@1':>9}{'n':>6}")
    for ck in sorted(total):
        acc = correct[ck] / total[ck]
        print(f"{ck:<40}{100*acc:>8.1f}%{total[ck]:>6}")
        rows.append({"config_key": ck, "accuracy": acc, "n": total[ck],
                     "correct": correct[ck], "metric": "humaneval"})

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"\nscores -> {args.out}")


if __name__ == "__main__":
    main()
