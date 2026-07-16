"""External AR anchor: run the SAME-input per-layer token-structure probe on a STANDALONE
causal LM, to check whether the TwoTower context tower's AR curve matches the public base
model it was initialized from (nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 — same 25T-token
init, same 52-layer/23-Mamba/23-MoE/6-attn arch as the context tower).

If nano_collapse ~= ctx_clean_collapse, the AR curve in layer_sim_4way is NOT a two-tower
coupling artifact — it's the intrinsic behaviour of the base model. That closes the last
"is the baseline itself weird?" question.

Single model — fits on ONE 80GB A100 (bf16 ~60GB weights). Reuses the exact prompts + metrics
from layer_sim_4way so the comparison is apples-to-apples.

    python src/layer_external_anchor.py --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
        --out results/nano_anchor.npz --ref results/layer_sim_4way.npz --plot
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer_sim_4way import BUILTIN_PROMPTS, cosine_matrix, offdiag_mean, diag_offset_mean, structure_of


def anchor_curves(model, tok, prompts, max_len):
    device = next(model.parameters()).device
    collapse_rows, adjacent_rows, heat0 = [], [], None
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
            out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
            hs = [h[0].detach().float().cpu() for h in out.hidden_states]
            col, adj = structure_of(hs)
            collapse_rows.append(col)
            adjacent_rows.append(adj)
            if i == 0:
                heat0 = cosine_matrix(hs[min(26, len(hs) - 1)]).numpy()
            print(f"[anchor] prompt {i + 1}/{len(prompts)} done  (layers={len(hs)})", flush=True)
    collapse = np.nanmean(np.array(collapse_rows, dtype=float), axis=0)
    adjacent = np.nanmean(np.array(adjacent_rows, dtype=float), axis=0)
    return collapse, adjacent, heat0


def make_plot(collapse, adjacent, ref, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = range(len(collapse))
    axes[0].plot(x, collapse, "-o", ms=2.5, color="#9467bd", label="Nano base (external AR)")
    axes[1].plot(x, adjacent, "-o", ms=2.5, color="#9467bd", label="Nano base (external AR)")
    if ref is not None:
        cc = ref.get("ctx_clean_collapse"); ca = ref.get("ctx_clean_adjacent")
        dc = ref.get("den_clean_collapse")
        if cc is not None:
            axes[0].plot(range(len(cc)), cc, "-o", ms=2.5, color="#1f77b4",
                         label="ctx_clean (TwoTower AR tower)")
        if ca is not None:
            axes[1].plot(range(len(ca)), ca, "-o", ms=2.5, color="#1f77b4",
                         label="ctx_clean (TwoTower AR tower)")
        if dc is not None:
            axes[0].plot(range(len(dc)), dc, "-o", ms=2.5, color="#d62728", alpha=0.6,
                         label="den_clean (diffusion tower)")
    axes[0].set_title("token collapse (mean off-diag cosine)"); axes[0].set_ylabel("cosine")
    axes[1].set_title("local smoothness (adjacent-token cosine)")
    for ax in axes:
        ax.set_xlabel("layer (0=embeddings .. 52)"); ax.legend(fontsize=7)
    fig.suptitle("External AR anchor: standalone Nano base vs TwoTower context tower")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"[plot] wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16")
    ap.add_argument("--out", default="results/nano_anchor.npz")
    ap.add_argument("--ref", help="layer_sim_4way.npz to overlay ctx_clean/den_clean")
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f">>> loading {args.model} (bf16, single GPU)")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True).to("cuda:0").eval()

    prompts = [t for _, t in BUILTIN_PROMPTS][:args.num_prompts]
    collapse, adjacent, heat0 = anchor_curves(model, tok, prompts, args.max_len)

    ref = None
    if args.ref and os.path.exists(args.ref):
        ref = np.load(args.ref)
        cc = ref.get("ctx_clean_collapse")
        if cc is not None and len(cc) == len(collapse):
            diff = float(np.nanmean(np.abs(np.asarray(cc) - collapse)))
            pla_nano = float(np.nanmean(collapse[8:41]))
            pla_ctx = float(np.nanmean(np.asarray(cc)[8:41]))
            print(f"\n=== anchor check (plateau L8..40) ===")
            print(f"  Nano base   plateau = {pla_nano:.3f}")
            print(f"  ctx_clean   plateau = {pla_ctx:.3f}")
            print(f"  mean |Nano - ctx_clean| per layer = {diff:.3f}")
            print("  >>> small gap (<0.05) => AR curve is the base model's intrinsic behaviour, "
                  "NOT a two-tower artifact.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save = {"nano_collapse": collapse, "nano_adjacent": adjacent}
    if heat0 is not None:
        save["nano_heat0"] = heat0
    np.savez(args.out, **save)
    print(f"[save] wrote {args.out}")

    if args.plot:
        make_plot(collapse, adjacent, ref, "figs/fig_nano_anchor.png")


if __name__ == "__main__":
    main()
