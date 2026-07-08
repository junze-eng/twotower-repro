"""Plot LOCALLY (Mac). Reads run_all jsonl (speed) and optional scores jsonl (quality),
produces the presentation figures. Config fields are parsed back out of config_key.

    python src/plot.py --speed results/e1.jsonl --kind speed_surface --out results/e1_surface.png
    python src/plot.py --speed results/e2.jsonl --scores results/e2_scores.jsonl --kind pareto --out results/pareto.png
    python src/plot.py --speed results/e3.jsonl --scores results/e3_scores.jsonl --kind collapse --out results/collapse.png
"""
import argparse
import json
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_ck(ck):
    """config_key like 'block_size16_gamma0.8_max_new256_steps16_temperature0.0'."""
    out = {}
    for key in ("block_size", "gamma", "steps", "max_new", "temperature"):
        m = re.search(rf"{key}(-?[\d.]+)", ck)
        if m:
            out[key] = float(m.group(1))
    return out


def load_speed(path):
    """Mean tps / tokens_per_nfe / nfe per config_key."""
    agg = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ck = r["config_key"]
            for k in ("tps", "tokens_per_nfe", "nfe", "wall_s"):
                if r.get(k) is not None:
                    agg[ck][k].append(r[k])
    return {ck: {k: float(np.mean(v)) for k, v in d.items()} for ck, d in agg.items()}


def load_scores(path):
    scores = {}
    if path:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                scores[r["config_key"]] = r["accuracy"]
    return scores


def speed_surface(speed, out):
    gammas = sorted({parse_ck(ck).get("gamma") for ck in speed})
    steps = sorted({parse_ck(ck).get("steps") for ck in speed})
    grid = np.full((len(gammas), len(steps)), np.nan)
    for ck, d in speed.items():
        p = parse_ck(ck)
        grid[gammas.index(p["gamma"]), steps.index(p["steps"])] = d["tokens_per_nfe"]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(steps))); ax.set_xticklabels([int(s) for s in steps])
    ax.set_yticks(range(len(gammas))); ax.set_yticklabels(gammas)
    ax.set_xlabel("steps_per_block (T)"); ax.set_ylabel("confidence_threshold (gamma)")
    ax.set_title("parallelism (tokens / NFE)")
    for i in range(len(gammas)):
        for j in range(len(steps)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i,j]:.1f}", ha="center", va="center", color="w")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out, dpi=120)
    print("wrote", out)


def pareto(speed, scores, out):
    xs, ys, labels = [], [], []
    for ck, acc in scores.items():
        if ck in speed:
            xs.append(speed[ck]["tps"]); ys.append(100 * acc); labels.append(ck)
    order = np.argsort(xs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.array(xs)[order], np.array(ys)[order], "o-", color="#1f77b4")
    for x, y, lab in zip(xs, ys, labels):
        p = parse_ck(lab)
        ax.annotate(f"g{p.get('gamma')}/T{int(p.get('steps',0))}", (x, y),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("throughput (tokens/sec)"); ax.set_ylabel("accuracy (%)")
    ax.set_title("quality vs speed Pareto"); fig.tight_layout(); fig.savefig(out, dpi=120)
    print("wrote", out)


def collapse(speed, scores, out):
    pts = []
    for ck, acc in scores.items():
        pts.append((parse_ck(ck).get("block_size"), 100 * acc))
    pts.sort()
    xs = [str(int(b)) for b, _ in pts]
    ys = [a for _, a in pts]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(xs, ys, color=["#2ca02c" if int(x) <= 16 else "#d62728" for x in xs])
    ax.set_xlabel("sampling block_size (trained at 16)"); ax.set_ylabel("accuracy (%)")
    ax.set_title("block-size collapse (Table 4)")
    ax.axvline(x=0.5 + xs.index("16") if "16" in xs else 0, color="gray", ls="--", lw=1)
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", required=True)
    ap.add_argument("--scores", default=None)
    ap.add_argument("--kind", choices=["speed_surface", "pareto", "collapse"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    speed = load_speed(args.speed)
    scores = load_scores(args.scores)
    if args.kind == "speed_surface":
        speed_surface(speed, args.out)
    elif args.kind == "pareto":
        pareto(speed, scores, args.out)
    else:
        collapse(speed, scores, args.out)


if __name__ == "__main__":
    main()
