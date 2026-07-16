"""Per-layer TOKEN-PAIR structure probe (AR tower vs diffusion tower), SAME real input.

Motivation / why this exists separately from layer_similarity.py
----------------------------------------------------------------
layer_similarity.py measures adjacent-LAYER redundancy, but it probes each tower in its
own regime: the AR tower sees the real prompt tokens while the denoiser sees a 16-token
all-[MASK] block. That confounds "training objective" with "degenerate all-MASK input"
(uniform input barely moves in early layers -> artificially high early-layer cosine). So
the AR-vs-diffusion early-layer gap there is NOT clean.

THIS probe removes that confound: it feeds the SAME real prompt tokens through BOTH towers
as plain output_hidden_states forwards, and asks how the TOKEN-PAIR similarity structure
evolves with depth. Input is identical for the two towers, so the only variable is the
learned weights (frozen-AR context tower vs diffusion-trained denoiser tower). This is the
same apples-to-apples setup Goel et al. (arXiv 2603.07475) use across models (same text
through both), applied here to two towers of ONE architecture from ONE init.

CAVEAT (state it in the report): running the denoiser tower as a plain forward on real,
unmasked tokens means it runs WITHOUT its AdaLN time-conditioning and WITHOUT cross-attn to
a context cache -> mildly out-of-distribution for the diffusion tower. That is the standard
way these cross-model layer analyses are done, but it is a real limitation. The faithful
(conditioned, all-MASK) view lives in layer_similarity.py; the two are complementary.

What it computes, per layer L, from the token hidden states H_L (shape [T, D]):
  S_L[i,j] = cos(H_L[i], H_L[j])              # token x token cosine matrix
  * offdiag_mean  — mean over i!=j              (global token collapse / anisotropy)
  * adjacent_mean — mean over |i-j|==1          (local smoothness / neighbour similarity)
  * by_distance   — mean over pairs at |i-j|==d  (similarity-vs-distance = locality/recency)
Curves are averaged over prompts; heatmaps of S_L at a few layers are dumped for prompt 0.

    python src/layer_token_structure.py --selftest          # pure-math, no model/GPU
    python src/layer_token_structure.py --prompts data/gsm8k_mini.jsonl --num-prompts 8 --plot
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F


# ======================================================================
# Token-pair structure math (model-agnostic).
# ======================================================================
def cosine_matrix(hidden_1tD, token_slice):
    """hidden_1tD: [1, T, D] one layer's hidden states. Returns [T', T'] cosine matrix."""
    x = hidden_1tD[0, token_slice, :].float()
    if x.ndim == 1:
        x = x.unsqueeze(0)
    xn = F.normalize(x, dim=-1)
    return xn @ xn.T


def _offdiag_mean(S):
    T = S.shape[0]
    if T < 2:
        return float("nan")
    off = (S.sum() - torch.diagonal(S).sum()) / (T * T - T)
    return off.item()


def _diag_offset_mean(S, d):
    T = S.shape[0]
    if T <= d:
        return float("nan")
    return torch.diagonal(S, d).mean().item()


def by_distance(S, max_d):
    """mean cosine at each token distance 1..max_d (nan when too few pairs)."""
    return [_diag_offset_mean(S, d) for d in range(1, max_d + 1)]


class LayerAccum:
    """Accumulate per-layer scalar curves across prompts (mean over prompts)."""
    def __init__(self):
        self.offdiag = None
        self.adjacent = None
        self.n = 0

    def add(self, offdiag_per_layer, adjacent_per_layer):
        if self.offdiag is None:
            self.offdiag = [0.0] * len(offdiag_per_layer)
            self.adjacent = [0.0] * len(adjacent_per_layer)
        for i, v in enumerate(offdiag_per_layer):
            if math.isfinite(v):
                self.offdiag[i] += v
        for i, v in enumerate(adjacent_per_layer):
            if math.isfinite(v):
                self.adjacent[i] += v
        self.n += 1

    def means(self):
        n = max(1, self.n)
        return ([s / n for s in self.offdiag], [s / n for s in self.adjacent])


def write_curves(path, offdiag, adjacent):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "offdiag_mean", "adjacent_mean"])
        w.writeheader()
        for i, (o, a) in enumerate(zip(offdiag, adjacent)):
            w.writerow({"layer": i, "offdiag_mean": o, "adjacent_mean": a})


# ======================================================================
# Capture: SAME real tokens through BOTH towers (plain forwards).
# ======================================================================
def _tower_hidden_states(tower, prompt_ids):
    device = next(tower.parameters()).device
    out = tower(input_ids=prompt_ids.to(device), use_cache=False, output_hidden_states=True)
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise RuntimeError(f"{type(tower).__name__} did not return hidden_states")
    return [h.detach() for h in hs]  # (embed, l1..lN)


def probe_prompt(model, tok, prompt, args, acc, heat_store):
    prompt_ids = tok(prompt, return_tensors="pt").input_ids
    towers = {"context": model.context_tower, "denoiser": model.denoiser_tower}
    for name, tower in towers.items():
        hs = _tower_hidden_states(tower, prompt_ids)
        T = hs[0].shape[1]
        sl = slice(0, T)
        offdiag, adjacent = [], []
        for L, h in enumerate(hs):
            S = cosine_matrix(h, sl)
            offdiag.append(_offdiag_mean(S))
            adjacent.append(_diag_offset_mean(S, 1))
            # stash a few full matrices from prompt 0 for heatmaps + a distance profile
            if heat_store is not None and L in heat_store["layers"]:
                heat_store[name][L] = S.cpu()
                heat_store["dist"][name][L] = by_distance(S, min(T - 1, args.max_dist))
        acc[name].add(offdiag, adjacent)


# ======================================================================
# Plots.
# ======================================================================
def make_plots(curves, heat_store, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"context": ("#1f77b4", "context tower (AR, frozen)"),
              "denoiser": ("#d62728", "denoiser tower (diffusion-trained)")}

    # --- figure 1: per-layer offdiag + adjacent curves, both towers ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, (offdiag, adjacent) in curves.items():
        c, lbl = styles[name]
        xs = list(range(len(offdiag)))
        axes[0].plot(xs, offdiag, "-o", ms=3, color=c, label=lbl)
        axes[1].plot(xs, adjacent, "-o", ms=3, color=c, label=lbl)
    axes[0].set_title("global token collapse  (mean off-diagonal cosine)")
    axes[1].set_title("local smoothness  (adjacent-token cosine, |i-j|=1)")
    for ax in axes:
        ax.set_xlabel("layer (0=embeddings .. 52)")
        ax.set_ylabel("token-pair cosine")
        ax.legend(fontsize=8)
    fig.suptitle("Per-layer token-pair structure: AR tower vs diffusion tower (same real input)")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    p1 = os.path.join(out_dir, "fig_layer_token_structure.png")
    fig.savefig(p1, dpi=130)
    print(f"[plot] wrote {p1}")

    if heat_store is None:
        return

    # --- figure 2: token x token cosine heatmaps at selected layers (prompt 0) ---
    layers = sorted(heat_store["layers"])
    fig, axes = plt.subplots(2, len(layers), figsize=(3.4 * len(layers), 6.6))
    if len(layers) == 1:
        axes = axes.reshape(2, 1)
    for col, L in enumerate(layers):
        for row, name in enumerate(("context", "denoiser")):
            ax = axes[row, col]
            S = heat_store[name].get(L)
            if S is None:
                ax.axis("off"); continue
            im = ax.imshow(S.numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_title(f"{name}  L{L}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="token-token cosine")
    fig.suptitle("Token x token cosine at selected layers (prompt 0) — top: AR, bottom: diffusion")
    p2 = os.path.join(out_dir, "fig_layer_token_heatmap.png")
    fig.savefig(p2, dpi=130)
    print(f"[plot] wrote {p2}")

    # --- figure 3: similarity vs token distance at selected layers ---
    fig, axes = plt.subplots(1, len(layers), figsize=(3.4 * len(layers), 4), sharey=True)
    if len(layers) == 1:
        axes = [axes]
    for ax, L in zip(axes, layers):
        for name in ("context", "denoiser"):
            prof = heat_store["dist"][name].get(L)
            if prof is None:
                continue
            c, lbl = styles[name]
            ax.plot(range(1, len(prof) + 1), prof, "-", color=c, label=lbl)
        ax.set_title(f"L{L}", fontsize=9)
        ax.set_xlabel("token distance |i-j|")
    axes[0].set_ylabel("mean cosine")
    axes[0].legend(fontsize=7)
    fig.suptitle("Similarity vs token distance (locality / recency) — prompt 0")
    fig.tight_layout()
    p3 = os.path.join(out_dir, "fig_layer_token_distance.png")
    fig.savefig(p3, dpi=130)
    print(f"[plot] wrote {p3}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", help="jsonl with a 'prompt' field")
    ap.add_argument("--prompt", help="a single literal prompt string (overrides --prompts)")
    ap.add_argument("--out", default="results/layer_token")
    ap.add_argument("--label", default="twotower")
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--max-dist", type=int, default=64,
                    help="cap on token distance for the by-distance/heatmap profiles")
    ap.add_argument("--heat-layers", default="4,26,50",
                    help="comma layers to dump full cosine heatmaps for (prompt 0)")
    ap.add_argument("--single", action="store_true", help="both towers on cuda:0")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from twotower import load
    from main_run import load_prompts

    if args.prompt:
        prompts = [{"prompt": args.prompt}]
    else:
        prompts = load_prompts(args.prompts, args.num_prompts)

    model, tok = load(den_device="cuda:0" if args.single else "cuda:1")
    acc = {"context": LayerAccum(), "denoiser": LayerAccum()}
    heat_layers = [int(x) for x in args.heat_layers.split(",") if x.strip() != ""]
    heat_store = {"layers": set(heat_layers), "context": {}, "denoiser": {},
                  "dist": {"context": {}, "denoiser": {}}}

    t0 = time.time()
    with torch.no_grad():
        for i, p in enumerate(prompts):
            probe_prompt(model, tok, p["prompt"], args,
                         acc, heat_store if i == 0 else None)
            print(f"[{args.label}] prompt {i + 1}/{len(prompts)} done", flush=True)

    curves = {name: a.means() for name, a in acc.items()}
    for name, (offdiag, adjacent) in curves.items():
        write_curves(f"{args.out}/{args.label}_{name}_token_structure.csv", offdiag, adjacent)

    def band(vals):
        v = [x for x in vals if math.isfinite(x)]
        return {"min": min(v), "max": max(v)} if v else {}

    summary = {"label": args.label, "num_prompts": len(prompts),
               "elapsed_sec": round(time.time() - t0, 1),
               "note": "SAME real tokens through both towers (plain forward); "
                       "denoiser runs w/o AdaLN/cross-attn (mild OOD).",
               "offdiag_mean": {n: band(c[0]) for n, c in curves.items()},
               "adjacent_mean": {n: band(c[1]) for n, c in curves.items()}}
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/{args.label}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.plot:
        make_plots(curves, heat_store, "figs")


# ----------------------------------------------------------------------
def _selftest():
    """No-model check: identical tokens -> cosine 1 everywhere; orthogonal tokens -> ~0;
    a local ramp -> adjacent cosine decays with distance."""
    torch.manual_seed(0)
    T, D = 8, 32
    same = torch.ones(1, T, D)
    S = cosine_matrix(same, slice(0, T))
    assert abs(_offdiag_mean(S) - 1.0) < 1e-5, _offdiag_mean(S)
    assert abs(_diag_offset_mean(S, 1) - 1.0) < 1e-5

    eye = torch.eye(T).unsqueeze(0)              # each token orthogonal to the others
    S2 = cosine_matrix(eye.float(), slice(0, T))
    assert abs(_offdiag_mean(S2)) < 1e-6, _offdiag_mean(S2)

    # smooth ramp: adjacent tokens more similar than far tokens -> by_distance decreasing
    ramp = (torch.linspace(0, 1, T).view(1, T, 1) + 0.1) * torch.ones(1, T, D)
    Sr = cosine_matrix(ramp, slice(0, T))
    prof = by_distance(Sr, T - 1)
    assert prof[0] >= prof[-1], prof
    acc = LayerAccum()
    acc.add([_offdiag_mean(S)], [_diag_offset_mean(S, 1)])
    o, a = acc.means()
    assert len(o) == 1 and abs(o[0] - 1.0) < 1e-5
    print(f"[selftest] OK  same_offdiag={_offdiag_mean(S):.4f}(==1) "
          f"orth_offdiag={_offdiag_mean(S2):.2e}(~0) "
          f"ramp dist {prof[0]:.3f}->{prof[-1]:.3f}(decays)")


if __name__ == "__main__":
    main()
