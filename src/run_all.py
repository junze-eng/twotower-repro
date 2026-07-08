"""The GPU workhorse: load the model ONCE, run a whole experiment's generations, append
each result to a jsonl. Scoring and plotting are separate LOCAL steps that read the jsonl.

Design:
  - one process, one model load -> pure generation time on the pod, no per-config reload
  - resumable: re-running skips (prompt_id, config_key) already in the output jsonl
  - records raw output text + NFE + tokens/NFE + wall_clock; NO quality scoring here

Experiments (config presets below):
  e1  speed surface   : gamma x steps grid (no scoring needed downstream)
  e3  collapse        : sampling block sweep {8,16,32,64} vs training block 16
  e2  pareto          : hand-picked representative (gamma, steps) points
  ar  baseline        : single-tower autoregressive

Prompts come from a jsonl file ({"prompt": ..., optional "id"/"task_id"/"reference"}),
so the same driver serves speed prompts, GSM8K, or HumanEval — the benchmark specifics
live in the prompt file, not here.

    python src/run_all.py --exp e1 --prompts data/speed_prompts.jsonl --out results/e1.jsonl
"""
import argparse
import json
import os
import time

import torch

from twotower import load, MASK_TOKEN_ID, reset_nfe, get_nfe


def diff_grid(max_new, block, gammas, steps_list):
    return [dict(mode="diff", max_new=max_new, block_size=block, steps=T, gamma=g,
                 temperature=0.0)
            for g in gammas for T in steps_list]


EXPERIMENTS = {
    # E1: speed surface. block fixed at training size; sweep gamma x steps.
    "e1": diff_grid(128, 16, (0.5, 0.7, 0.8, 0.9, 0.95), (4, 8, 16)),
    # E3: collapse. sampling block sweep; 256 is divisible by all of 8/16/32/64.
    "e3": [dict(mode="diff", max_new=256, block_size=B, steps=16, gamma=0.8,
                temperature=0.0) for B in (8, 16, 32, 64)],
    # E2: pareto. representative points spanning fast->slow (refine from E1 surface).
    "e2": [dict(mode="diff", max_new=256, block_size=16, steps=T, gamma=g, temperature=0.0)
           for (g, T) in [(0.5, 4), (0.7, 8), (0.8, 8), (0.8, 16), (0.9, 16), (0.95, 16)]],
    # AR baseline (single tower).
    "ar": [dict(mode="ar", max_new=256)],
}


def config_key(cfg):
    return "_".join(f"{k}{cfg[k]}" for k in sorted(cfg) if k != "mode")


def run_one(model, tok, prompt, cfg):
    input_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")
    plen = input_ids.shape[1]
    reset_nfe(model)
    t0 = time.time()
    with torch.no_grad():
        if cfg["mode"] == "ar":
            out = model.generate_ar(input_ids, max_new_tokens=cfg["max_new"])
            nfe = cfg["max_new"]                      # AR: one forward per token
        else:
            out = model.generate_mask_diffusion(
                input_ids, max_new_tokens=cfg["max_new"],
                block_size=cfg["block_size"], steps_per_block=cfg["steps"],
                mask_token_id=MASK_TOKEN_ID, temperature=cfg["temperature"],
                confidence_threshold=cfg["gamma"], eos_token_id=tok.eos_token_id,
            )
            nfe = get_nfe(model)
    dt = time.time() - t0
    text = tok.decode(out[0][plen:], skip_special_tokens=True)
    return dict(output=text, nfe=nfe,
                tokens_per_nfe=(cfg["max_new"] / nfe) if nfe else None,
                tps=round(cfg["max_new"] / dt, 2),   # tokens/sec — the hardware-dependent 2.42x metric
                wall_s=round(dt, 3))


def load_prompts(path, limit):
    items = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d.setdefault("id", d.get("task_id", str(i)))
            items.append(d)
    return items[:limit] if limit else items


def done_keys(out_path):
    keys = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    keys.add((r["prompt_id"], r["config_key"]))
                except Exception:
                    pass
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(EXPERIMENTS))
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap number of prompts (0=all)")
    args = ap.parse_args()

    cfgs = EXPERIMENTS[args.exp]
    prompts = load_prompts(args.prompts, args.limit)
    done = done_keys(args.out)
    ar_only = all(c["mode"] == "ar" for c in cfgs)

    print(f"exp={args.exp}  configs={len(cfgs)}  prompts={len(prompts)}  "
          f"already_done={len(done)}  ar_only={ar_only}")
    model, tok = load(ar_only=ar_only)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    total = len(cfgs) * len(prompts)
    n = 0
    with open(args.out, "a", encoding="utf-8") as fout:
        for cfg in cfgs:
            ck = config_key(cfg)
            for p in prompts:
                n += 1
                if (p["id"], ck) in done:
                    continue
                r = run_one(model, tok, p["prompt"], cfg)
                rec = dict(exp=args.exp, prompt_id=p["id"], config_key=ck,
                           **{k: cfg[k] for k in cfg}, **r)
                for extra in ("task_id", "reference"):
                    if extra in p:
                        rec[extra] = p[extra]
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"[{n}/{total}] {ck} id={p['id']} "
                      f"nfe={r['nfe']} tpn={r['tokens_per_nfe']} {r['wall_s']}s")
    print("done ->", args.out)


if __name__ == "__main__":
    main()
