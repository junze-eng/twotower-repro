"""Build mini benchmark prompt files. Works OFFLINE.

  gsm8k     : 4-shot chain-of-thought, reference = gold final number.
              Tries HF `datasets` (real GSM8K); if offline/unavailable, falls back to an
              embedded set of simple arithmetic word problems (answers authored+verified here)
              so it works with HF_HUB_OFFLINE=1. The embedded set is labeled synthmath_* and is
              NOT official GSM8K — fine for a quality/trend smoke, note it in the writeup.
  humaneval : 0-shot completion (needs HF datasets / network).

    python src/prep_data.py --which gsm8k     --n 15 --out data/gsm8k_mini.jsonl
    python src/prep_data.py --which humaneval --n 25 --out data/humaneval_mini.jsonl
"""
import argparse
import json
import os

GSM8K_SHOTS = """Question: Natalia sold clips to 48 friends in April, and half as many in May. How many clips did she sell altogether?
Answer: In May she sold 48 / 2 = 24 clips. Altogether 48 + 24 = 72. #### 72

Question: Weng earns $12 an hour for babysitting. Yesterday she babysat 50 minutes. How much did she earn?
Answer: Per minute she earns 12 / 60 = $0.2. For 50 minutes she earned 50 * 0.2 = $10. #### 10

Question: Betty has half the money she needs for a $100 wallet. Her parents give her $15 and her grandparents twice as much. How much more does she need?
Answer: Betty has 100 / 2 = $50. Grandparents give 15 * 2 = $30. Now she has 50 + 15 + 30 = $95. She needs 100 - 95 = $5. #### 5

Question: James writes a 3-page letter to 2 friends twice a week. How many pages does he write a year?
Answer: Each time he writes 3 * 2 = 6 pages. Twice a week that's 6 * 2 = 12 pages. In a year 12 * 52 = 624. #### 624

"""

# Authored + verified simple arithmetic word problems (offline fallback). answers are correct.
_EMBEDDED = [
    ("Tom has 12 apples. He buys 8 more and then gives 5 to his friend. How many apples does he have now?", "15"),
    ("A book has 240 pages. Sarah reads 60 pages each day. How many days does she need to finish it?", "4"),
    ("There are 5 boxes with 12 pencils each. How many pencils are there in total?", "60"),
    ("Maria earns $15 per hour and works 6 hours. How much does she earn?", "90"),
    ("A train travels 80 km in 2 hours. What is its speed in km per hour?", "40"),
    ("John had 100 dollars. He spent 35 on a shirt and 20 on lunch. How much money is left?", "45"),
    ("A class has 30 students. 18 of them are girls. How many are boys?", "12"),
    ("A rectangle is 8 meters long and 3 meters wide. What is its area in square meters?", "24"),
    ("Anna bakes 4 dozen cookies. How many cookies is that?", "48"),
    ("A car uses 6 liters of fuel per 100 km. How many liters does it use for 250 km?", "15"),
    ("Ben saves $7 each week. How much does he save in 8 weeks?", "56"),
    ("A pizza is cut into 8 slices. 3 people each eat 2 slices. How many slices are left?", "2"),
    ("There are 3 shelves with 25 books each. How many books are there in total?", "75"),
    ("Lucy has 45 stickers. She gives away 18 and then buys 10 more. How many stickers does she have now?", "37"),
    ("A farmer has 9 cows and each cow gives 12 liters of milk. How many liters of milk in total?", "108"),
]


def build_gsm8k(n, out):
    items = []
    src = "gsm8k(HF)"
    try:
        from datasets import load_dataset
        try:
            ds = load_dataset("gsm8k", "main", split="test")
        except Exception:
            ds = load_dataset("openai/gsm8k", "main", split="test")
        for i in range(min(n, len(ds))):
            gold = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
            items.append((f"gsm8k_{i}", ds[i]["question"], gold))
    except Exception as e:
        print(f"[prep] HF datasets unavailable ({type(e).__name__}); using embedded synthetic set")
        src = "embedded-synthetic"
        for i, (q, a) in enumerate(_EMBEDDED[:n]):
            items.append((f"synthmath_{i}", q, a))

    with open(out, "w", encoding="utf-8") as f:
        for id_, q, gold in items:
            prompt = GSM8K_SHOTS + f"Question: {q}\nAnswer:"
            f.write(json.dumps({"id": id_, "prompt": prompt, "reference": gold},
                               ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} prompts from {src} -> {out}")


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
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    (build_gsm8k if args.which == "gsm8k" else build_humaneval)(args.n, args.out)


if __name__ == "__main__":
    main()
