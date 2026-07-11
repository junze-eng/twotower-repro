"""analyze_commit.py (LOCAL, no GPU) — parallelism / commit-order evidence from an
exp0_capture frames npz. Produces:
  - Kendall tau-b of (reply position vs first-commit step)  -> left-to-right bias (cf. the
    DiffusionGemma "Neither Parallel Nor Sequential" paper, which reports tau-b 0.43-0.60)
  - per-step commit-batch size distribution                -> parallelism WIDTH (cf. DG's
    "accept batch 13-26 tokens")
  - the TRUE position x step triangle heatmap
  - the TRUE per-position denoising GIF

    python analyze_commit.py --npz results/trace_tri_b16.npz --outdir figs

Reconstruction: the capture stores each step's raw denoiser canvas row (block-local ~16-wide,
or cumulative) plus block_idx / frame_width, so we place every row at the right reply offset
and carry committed tokens forward. A position "commits" the first step its value != MASK.
"""
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

BLUE, GREEN, ORANGE, VIOLET, RED = "#2a78d6", "#008300", "#eb6834", "#4a3aa7", "#e34948"
MASKC = (0.91, 0.91, 0.90)
INK, INK2, MUTED, GRID, SURF, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]


def reconstruct(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    frames = d["frames"]; meta = json.loads(str(d["meta"]))
    mask_id = int(meta["mask_token_id"]); bs = int(meta["block_size"]); mx = int(meta["max_new"])
    plen = int(meta["prompt_len"]); blocks = list(meta["block_idx"])
    widths = list(meta.get("frame_width") or [frames.shape[1]] * frames.shape[0])
    F = frames.shape[0]
    canvas = np.full((F, mx), mask_id, np.int32)          # carry-forward reply canvas per step
    running = np.full(mx, mask_id, np.int32)
    layout = None
    for f in range(F):
        w = int(widths[f]); row = frames[f, :w]; b = int(blocks[f])
        if w == bs:
            off = b * bs; layout = layout or "block-local"
        elif w <= mx:
            off = 0; layout = layout or "cumulative"
        else:
            row = row[plen:]; off = 0; layout = layout or "prompt-included"
        seg = row[:max(0, mx - off)]
        running[off:off + seg.shape[0]] = seg            # live state incl. remask
        canvas[f] = running
    print(f"[reconstruct] F={F} layout={layout} max_new={mx} block_size={bs} mask_id={mask_id}")
    return canvas, meta, mask_id


def kendall_tau_b(x, y):
    n = len(x); nc = nd = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]; dy = y[i] - y[j]
            if dx == 0: tx += 1
            if dy == 0: ty += 1
            s = (dx > 0) - (dx < 0)
            t = (dy > 0) - (dy < 0)
            if s * t > 0: nc += 1
            elif s * t < 0: nd += 1
    n0 = n * (n - 1) / 2
    denom = np.sqrt((n0 - tx) * (n0 - ty))
    return (nc - nd) / denom if denom > 0 else float("nan")


def analyze(npz_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    canvas, meta, mask_id = reconstruct(npz_path)
    F, mx = canvas.shape
    committed = canvas != mask_id                          # (F, mx) bool

    # first-commit step per position (positions never committed -> excluded)
    first = np.full(mx, -1, np.int64)
    for p in range(mx):
        nz = np.nonzero(committed[:, p])[0]
        if nz.size: first[p] = nz[0]
    pos = np.nonzero(first >= 0)[0]
    tau = kendall_tau_b(pos.tolist(), first[pos].tolist()) if pos.size > 2 else float("nan")

    # per-step commit-batch: positions newly committed (mask -> non-mask) at each step
    newc = committed & ~np.vstack([np.zeros((1, mx), bool), committed[:-1]])
    batch = newc.sum(1)
    batch_nz = batch[batch > 0]
    stats = dict(
        tau_b=round(float(tau), 3),
        n_committed=int(pos.size),
        commit_batch_mean=round(float(batch_nz.mean()), 2) if batch_nz.size else 0,
        commit_batch_median=int(np.median(batch_nz)) if batch_nz.size else 0,
        commit_batch_max=int(batch_nz.max()) if batch_nz.size else 0,
        n_steps=int(F), nfe=meta.get("nfe"),
    )
    print("[stats]", json.dumps(stats, ensure_ascii=False))

    # ---- FIG: true triangle (fill history) + first-commit scatter with tau ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.patch.set_facecolor(SURF)
    a1.imshow(committed, aspect="auto", cmap="Blues", interpolation="nearest",
              extent=[0, mx, F, 0], vmin=0, vmax=1)
    a1.set_xlabel("reply position", color=INK2, fontsize=10)
    a1.set_ylabel("denoising step (time →down)", color=INK2, fontsize=10)
    a1.set_title("Commit history (upper-left triangle = left-to-right)", color=INK,
                 fontsize=11, fontweight="bold", loc="left")
    a2.set_facecolor(SURF)
    for s in ("top", "right"): a2.spines[s].set_visible(False)
    a2.scatter(pos, first[pos], color=BLUE, s=24, zorder=3)
    a2.set_xlabel("reply position", color=INK2, fontsize=10)
    a2.set_ylabel("first-commit step", color=INK2, fontsize=10)
    a2.set_title(f"Left-to-right bias: Kendall τb = {stats['tau_b']}", color=INK,
                 fontsize=11, fontweight="bold", loc="left")
    a2.grid(color=GRID, lw=0.8); a2.set_axisbelow(True); a2.tick_params(colors=MUTED)
    a2.annotate("1.0 = strict AR · 0 = order-independent", (0.02, 0.94), xycoords="axes fraction",
                color=MUTED, fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_commit_order.png"), dpi=140, facecolor=SURF)
    plt.close(fig)

    # ---- FIG: commit-batch distribution ----
    fig, ax = plt.subplots(figsize=(7, 4.4)); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if batch_nz.size:
        ax.hist(batch_nz, bins=range(1, int(batch_nz.max()) + 2), color=VIOLET, align="left", rwidth=0.85)
    ax.axvline(stats["commit_batch_mean"], color=ORANGE, ls="--", lw=1.6)
    ax.annotate(f"mean {stats['commit_batch_mean']} tokens/step (max {stats['commit_batch_max']})",
                (stats["commit_batch_mean"], 0), xytext=(6, 20), textcoords="offset points",
                color=ORANGE, fontsize=9)
    ax.set_xlabel("content tokens committed in one forward pass", color=INK2, fontsize=10)
    ax.set_ylabel("# denoising steps", color=INK2, fontsize=10)
    ax.set_title("Parallelism width: commit-batch per step", color=INK, fontsize=11.5,
                 fontweight="bold", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(colors=MUTED)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_commit_batch.png"), dpi=140, facecolor=SURF)
    plt.close(fig)

    # ---- TRUE denoising GIF (real per-position commits) ----
    nrows = int(np.ceil(mx / meta["block_size"])); bs = int(meta["block_size"])
    commit_step = np.where(first >= 0, first, F + 1)
    gframes = []
    for f in range(F):
        img = np.ones((nrows, bs, 3))
        for p in range(mx):
            r, c = divmod(p, bs)
            if not committed[f, p]:
                img[r, c] = MASKC
            else:
                age = f - commit_step[p]
                img[r, c] = (0.0, 0.62, 0.31) if age == 0 else (0.92, 0.41, 0.20) if age <= 2 else (0.165, 0.47, 0.84)
        fig, ax = plt.subplots(figsize=(6, 6.4)); fig.patch.set_facecolor(SURF)
        ax.imshow(img, interpolation="nearest", extent=[0, bs, nrows, 0])
        for k in range(nrows + 1): ax.axhline(k, color=SURF, lw=1.5)
        for k in range(bs + 1): ax.axvline(k, color=SURF, lw=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Denoising (measured per-position commits)", color=INK, fontsize=12, fontweight="bold", loc="left")
        ax.set_xlabel(f"step {f+1}/{F}   ·   committed {int(committed[f].sum())}/{mx}   ·   τb={stats['tau_b']}",
                      color=INK2, fontsize=10)
        fig.tight_layout(); p_ = os.path.join(outdir, f"_rf_{f:03d}.png")
        fig.savefig(p_, dpi=88, facecolor=SURF); plt.close(fig); gframes.append(imageio.imread(p_))
    imageio.mimsave(os.path.join(outdir, "denoising_real.gif"), gframes + [gframes[-1]] * 8, fps=6, loop=0)
    for f in range(F):
        fp = os.path.join(outdir, f"_rf_{f:03d}.png")
        if os.path.exists(fp): os.remove(fp)

    with open(os.path.join(outdir, "commit_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    print("wrote fig_commit_order.png, fig_commit_batch.png, denoising_real.gif, commit_stats.json ->", outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/trace_tri_b16.npz")
    ap.add_argument("--outdir", default="figs")
    a = ap.parse_args()
    analyze(a.npz, a.outdir)
