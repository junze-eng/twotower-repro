"""Exp0/AnalysisA (LOCAL, no GPU): quantify the denoising dynamics from a trace.npz.

Directly tests the hypothesis "most tokens get committed in the first step, so it runs
fast". Produces the numbers behind the upper-left triangle:
  - commits-per-step and remasks-per-step
  - fraction of each block committed already at its FIRST step
  - cumulative fill curve
  - effective NFE vs the max (steps_per_block * blocks) -> is there early-stop headroom?

Run on your Mac/laptop:
    python src/exp0_analyze.py --npz results/trace.npz --out results/exp0_dynamics.png
"""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/trace.npz")
    ap.add_argument("--out", default="results/exp0_dynamics.png")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    frames = d["frames"]                     # (F, L)
    meta = json.loads(str(d["meta"]))
    mask_id = meta["mask_token_id"]
    block_idx = np.asarray(meta["block_idx"])  # per-frame block id
    step_idx = np.asarray(meta["step_idx"])    # per-frame step within block
    F, L = frames.shape
    is_mask = frames == mask_id

    # per-step deltas (a position going mask->token = commit, token->mask = remask)
    commits = np.zeros(F, int)
    remasks = np.zeros(F, int)
    for f in range(1, F):
        commits[f] = int((is_mask[f - 1] & ~is_mask[f]).sum())
        remasks[f] = int((~is_mask[f - 1] & is_mask[f]).sum())
    filled = (~is_mask).sum(axis=1)          # cumulative filled per frame

    # fraction of each block committed at its FIRST step
    print(f"\nsample: {meta['sample_id']}  block_size={meta['block_size']}  "
          f"steps={meta['steps']}  gamma={meta['gamma']}")
    print(f"frames(=NFE proxy)={F}  reply_len={L}  tokens/NFE={L / F:.2f}  "
          f"reported NFE={meta.get('nfe')}\n")
    print(f"{'block':>5}{'step1 commits':>15}{'block_size':>12}{'step1 %':>10}")
    for b in sorted(set(block_idx.tolist())):
        fs = np.where(block_idx == b)[0]
        first = fs[step_idx[fs].argmin()]
        c1 = commits[first] if first > 0 else int((~is_mask[first]).sum())
        bs = meta["block_size"]
        print(f"{b:>5}{c1:>15}{bs:>12}{100 * c1 / bs:>9.1f}%")

    max_nfe = meta["block_size"] and (L // meta["block_size"]) * meta["steps"]
    print(f"\nearly-stop check: frames={F} vs max possible steps={max_nfe} "
          f"-> {'headroom (early-stop or few steps used)' if F < max_nfe else 'ran full loop'}")
    print(f"total remask events: {int(remasks.sum())}")

    # figures
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[2, 1])
    ax[0].bar(range(F), commits, color="#2ca02c", label="commits")
    ax[0].bar(range(F), -remasks, color="#d62728", label="remasks")
    ax[0].set_title("commits (+) / remasks (-) per denoising step")
    ax[0].set_xlabel("step (frame)"); ax[0].legend()
    ax[1].plot(range(F), 100 * filled / L, color="#1f77b4")
    ax[1].set_title("cumulative fill %"); ax[1].set_xlabel("step"); ax[1].set_ylim(0, 101)
    fig.tight_layout(); fig.savefig(args.out, dpi=110)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
