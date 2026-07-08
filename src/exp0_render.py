"""Exp0 (LOCAL/offline, no GPU): render a captured trace.npz into a GIF like the mock.

Layout per frame:
  - title + sample id + "filled=x/N | completion=y%"
  - legend: cell color = how many frames ago this position was (re)written
            1st(current)=green, 2nd=orange, 3rd=blue, 4th=red, 5th+=purple; masked=grey
  - main grid: one cell per reply-span position (masked positions show grey)
  - bottom strip: fill-state history stacked by frame -> the upper-left triangle
  - frame counter f/F

Run this on your laptop; it only needs numpy + matplotlib + imageio.
    python src/exp0_render.py --npz results/trace.npz --out results/exp0.gif
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio.v2 as imageio

AGE_COLORS = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728", "#9467bd"]  # 1st..5th+
MASK_COLOR = "#e8e8e8"


def compute_age(frames, mask_id):
    """age[f,p] = frames since position p last (re)took its current non-mask value;
    -1 means currently masked (handles remask: value reverts to grey)."""
    F, L = frames.shape
    age = np.full((F, L), -1, np.int32)
    last_val = np.full(L, mask_id, frames.dtype)
    last_change = np.full(L, -1, np.int32)
    for f in range(F):
        row = frames[f]
        for p in range(L):
            v = row[p]
            if v == mask_id:
                last_val[p] = mask_id
                last_change[p] = -1
                age[f, p] = -1
            else:
                if v != last_val[p]:
                    last_change[p] = f
                    last_val[p] = v
                age[f, p] = f - last_change[p]
    return age


def render(npz_path, out_gif, cols=64, fps=8):
    d = np.load(npz_path, allow_pickle=True)
    frames = d["frames"]
    meta = json.loads(str(d["meta"]))
    mask_id = meta["mask_token_id"]
    F, L = frames.shape
    age = compute_age(frames, mask_id)
    fill_hist = (frames != mask_id).astype(np.float32)  # (F, L)
    rows = int(np.ceil(L / cols))
    os.makedirs("results/frames", exist_ok=True)

    images = []
    for f in range(F):
        fig = plt.figure(figsize=(9, 12))
        gs = fig.add_gridspec(2, 1, height_ratios=[6, 1], hspace=0.2)
        ax = fig.add_subplot(gs[0]); ax.set_xlim(0, cols); ax.set_ylim(-rows, 3); ax.axis("off")

        filled = int(fill_hist[f].sum())
        ax.text(0, 2.4, "Generated reply span", fontsize=20, fontweight="bold")
        ax.text(0, 1.8, f"{meta['sample_id']}/sample", fontsize=10, color="#444")
        ax.text(0, 1.4, f"filled={filled}/{L} | completion={100*filled/L:.1f}%",
                fontsize=10, color="#444")
        for i, lab in enumerate(["1st (current frame)", "2nd", "3rd", "4th", "5th+"]):
            ax.text(i * 13, 0.7, "● " + lab, color=AGE_COLORS[i], fontsize=9)

        for p in range(L):
            r, c = divmod(p, cols)
            col = MASK_COLOR if age[f, p] < 0 else AGE_COLORS[min(age[f, p], 4)]
            ax.add_patch(Rectangle((c, -r), 0.9, 0.9, color=col))

        axb = fig.add_subplot(gs[1])
        axb.imshow(fill_hist[:f + 1], aspect="auto", cmap="Greys", vmin=0, vmax=1,
                   interpolation="nearest", extent=[0, L, f + 1, 0])
        axb.set_title("Token positions (history kept below)", loc="left", fontsize=10)
        axb.set_yticks([])
        axb.set_xticks([0, L * .25, L * .5, L * .75, L])
        axb.set_xticklabels(["0", "25%", "50%", "75%", str(L)])
        fig.text(0.88, 0.02, f"{f + 1}/{F}", fontsize=12)

        path = f"results/frames/frame_{f:04d}.png"
        fig.savefig(path, dpi=90, bbox_inches="tight"); plt.close(fig)
        images.append(imageio.imread(path))

    # hold the last frame a bit longer
    imageio.mimsave(out_gif, images + [images[-1]] * fps, fps=fps)
    print(f"wrote {out_gif}  ({F} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/trace.npz")
    ap.add_argument("--out", default="results/exp0.gif")
    ap.add_argument("--cols", type=int, default=64)
    ap.add_argument("--fps", type=int, default=8)
    a = ap.parse_args()
    render(a.npz, a.out, cols=a.cols, fps=a.fps)
