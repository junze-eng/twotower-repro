"""Journal-style (Goel et al. Fig 1a) redraw of the token-structure figures from the
4-way probe's npz. Pure numpy + matplotlib — NO GPU, runs locally.

  fig_token_structure_journal.png  — (a) global collapse + (b) local smoothness,
                                     ctx_clean (solid black) vs den_clean (dashed gray).
  fig_token_heatmap_journal.png    — 2x3 token x token cosine: top context L4/26/50,
                                     bottom denoiser L4/26/50 (den_clean = SAME clean input,
                                     causal, no seed, no t — so the only difference is weights).
  fig_token_heatmap_4way_L26.png   — 1x4 at L26: ctx_clean / den_clean / den_bidir / den_natural,
                                     decomposing the collapse (weights -> +bidir -> +MASK/seed).

Denoiser rows use den_clean (NOT den_natural) so the input is identical to context; this closes
the "is it just because the input is all [MASK]?" objection.

    python make_journal_figs.py --npz results/layer_sim_4way.npz
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#FDF6E3"          # solarized-light beige; set --white for plain white
ATTN = [5, 12, 19, 26, 33, 42]
HEAT_LAYERS = [4, 26, 50]


def style(white):
    bg = "white" if white else BG
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.facecolor": bg, "figure.facecolor": bg, "savefig.facecolor": bg,
        "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    })
    return bg


def get_heat(npz, path, L):
    """Token x token matrix for `path` at hidden-state index L. Falls back to the old
    single-layer key `<path>_heat0` (which was captured at L26)."""
    k = f"{path}_heat_L{L}"
    if k in npz.files:
        return npz[k]
    if L == 26 and f"{path}_heat0" in npz.files:
        return npz[f"{path}_heat0"]
    return None


def fig_lines(npz, bg, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    series = [("ctx_clean", "black", "-", 1.8, "context tower (AR)"),
              ("den_clean", "#555555", "--", 1.5, "denoiser tower (diffusion)")]
    for ax, metric, ylab in [
        (axes[0], "collapse", r"$\mathrm{mean}_{i\neq j}\ \cos(h_i,\,h_j)$"),
        (axes[1], "adjacent", r"$\cos(h_i,\,h_{i+1})$")]:
        for path, color, ls, lw, lbl in series:
            y = npz[f"{path}_{metric}"]
            ax.plot(range(len(y)), y, ls=ls, lw=lw, color=color, label=lbl)
        for L in ATTN:
            ax.axvline(L, ls=":", lw=0.8, color="#999999", alpha=0.7)
        ax.grid(True, alpha=0.3, color="#999999", lw=0.5)
        ax.set_xlabel("layer"); ax.set_ylabel(ylab)
        ax.legend(frameon=False, loc="lower right")
    axes[0].set_title("(a) global token collapse")
    axes[1].set_title("(b) local smoothness")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"[plot] wrote {out}")


def _heatgrid(mats_by_pos, titles, suptitle, out, bg, ncols):
    nrows = len(mats_by_pos) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows + 0.4))
    axes = np.atleast_1d(axes).ravel()
    im = None
    for ax, (M, ttl) in zip(axes, zip(mats_by_pos, titles)):
        if M is None:
            ax.axis("off"); continue
        im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="Greys", aspect="equal")
        ax.set_title(ttl); ax.set_xticks([]); ax.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label=r"$\cos(h_i,\,h_j)$")
    fig.suptitle(suptitle, fontsize=10)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[plot] wrote {out}")


def fig_heatmap_2x3(npz, bg, out):
    mats, titles = [], []
    ok = True
    for path, tag in [("ctx_clean", "context"), ("den_clean", "denoiser")]:
        for L in HEAT_LAYERS:
            M = get_heat(npz, path, L)
            if M is None and L != 26:
                ok = False
            mats.append(M); titles.append(f"{tag}  L{L}")
    if not ok:
        print("[skip] 2x3 heatmap needs heat at L4/L26/L50 for ctx_clean+den_clean — "
              "re-run:  python src/layer_sim_4way.py --heat-layers 4,26,50 --out results/layer_sim_4way.npz")
        return
    _heatgrid(mats, titles,
              "Token×token cosine — same clean input, causal, no seed, no t  "
              "(top: AR tower, bottom: diffusion tower)",
              out, bg, ncols=3)


def fig_heatmap_4way(npz, bg, out):
    paths = ["ctx_clean", "den_clean", "den_bidir", "den_natural"]
    labels = ["ctx_clean\n(AR)", "den_clean\n(+weights)", "den_bidir\n(+bidir attn)",
              "den_natural\n(+MASK/seed)"]
    mats = [get_heat(npz, p, 26) for p in paths]
    if any(m is None for m in mats):
        print("[skip] 4-way L26 heatmap needs <path>_heat_L26 for all four paths.")
        return
    _heatgrid(mats, labels,
              "Token×token cosine at L26 — decomposing the collapse "
              "(weights → +bidirectional → +MASK/seed)",
              out, bg, ncols=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/layer_sim_4way.npz")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--white", action="store_true", help="plain white background instead of beige")
    args = ap.parse_args()

    bg = style(args.white)
    npz = np.load(args.npz)
    os.makedirs(args.outdir, exist_ok=True)
    fig_lines(npz, bg, os.path.join(args.outdir, "fig_token_structure_journal.png"))
    fig_heatmap_2x3(npz, bg, os.path.join(args.outdir, "fig_token_heatmap_journal.png"))
    fig_heatmap_4way(npz, bg, os.path.join(args.outdir, "fig_token_heatmap_4way_L26.png"))


if __name__ == "__main__":
    main()
