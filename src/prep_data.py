"""Build mini benchmark prompt files LOCALLY (Mac, needs `pip install datasets`).

Base model -> few-shot / completion prompting, no chat template.
  gsm8k    : 4-shot chain-of-thought, reference = gold final number
  humaneval: 0-shot completion, reference = {test, entry_point} for sandbox scoring

    python src/prep_data.py --which gsm8k     --n 25 --out data/gsm8k_mini.jsonl
    python src/prep_data.py --which humaneval --n 25 --out data/humaneval_mini.jsonl
"""
import argparse
import json
import os
import re

# compact standard GSM8K chain-of-thought exemplars (Wei et al. style)
GSM8K_SHOTS = """Question: Natalia sold clips to 48 friends in April, and half as many in May. How many clips did she sell altogether?
Answer: In May she sold 48 / 2 = 24 clips. Altogether 48 + 24 = 72. #### 72

Question: Weng earns $12 an hour for babysitting. Yesterday she babysat 50 minutes. How much did she earn?
Answer: Per minute she earns 12 / 60 = $0.2. For 50 minutes she earned 50 * 0.2 = $10. #### 10

Question: Betty has half the money she needs for a $100 wallet. Her parents give her $15 and her grandparents twice as much. How much more does she need?
Answer: Betty has 100 / 2 = $50. Grandparents give 15 * 2 = $30. Now she has 50 + 15 + 30 = $95. She needs 100 - 95 = $5. #### 5

Question: James writes a 3-page letter to 2 friends twice a week. How many pages does he write a year?
Answer: Each time he writes 3 * 2 = 6 pages. Twice a week that's 6 * 2 = 12 pages. In a year 12 * 52 = 624. #### 624

"""


def build_gsm8k(n, out):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    with open(out, "w", encoding="utf-8") as f:
        for i in range(min(n, len(ds))):
            q = ds[i]["question"]
            gold = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
            prompt = GSM8K_SHOTS + f"Question: {q}\nAnswer:"
            f.write(json.dumps({"id": f"gsm8k_{i}", "prompt": prompt,
                                "reference": gold}, ensure_ascii=False) + "\n")
    print(f"wrote {min(n, len(ds))} gsm8k prompts -> {out}")


def build_humaneval(n, out):
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    with open(out, "w", encoding="utf-8") as f:
        for i in range(min(n, len(ds))):
            ex = ds[i]
            f.write(json.dumps({"id": ex["task_id"], "task_id": ex["task_id"],
                                "prompt": ex["prompt"],
                                "reference": {"test": ex["test"],
                                              "entry_point": ex["entry_point"]}},
                               ensure_ascii=False) + "\n")
    print(f"wrote {min(n, len(ds))} humaneval prompts -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["gsm8k", "humaneval"], required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    (build_gsm8k if args.which == "gsm8k" else build_humaneval)(args.n, args.out)


if __name__ == "__main__":
    main()
