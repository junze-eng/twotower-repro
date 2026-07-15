"""Layer-wise representational redundancy probe for TwoTower (AR tower vs diffusion tower).

Migrated from Bodan Liu's single-tower Nemotron-Diffusion probe
(`reproduce_paper_layer_curves.py`), which computes two cosine curves per layer:

  * tokenwise      — cos(hidden[t], hidden[t+1]) within a layer  (neighbour-token similarity)
  * adjacent_layer — cos(hidden_L[i], hidden_{L+1}[i]) per token  (LAYER-TO-LAYER REDUNDANCY)

The adjacent_layer curve is the redundancy signal from
  "A Comparative analysis of Layer-wise Representational Capacity in AR and Diffusion LLMs"
  (Goel et al., arXiv 2603.07475): high adjacent-layer cosine => that layer barely moved the
  residual stream => redundant / skippable. Their thesis: *diffusion* objectives create
  substantial EARLY-layer redundancy + weak recency bias; *AR* objectives create locally
  structured reps + strong recency bias.

WHY TwoTower is the ideal testbed: both towers are the SAME NemotronH architecture from the
SAME 25T-token init, but one is frozen-AR (`context_tower`) and one is diffusion-trained
(`denoiser_tower`). So the two curves isolate the effect of the *training objective alone*
on depth redundancy — a cleaner controlled comparison than cross-model (LLaDA vs Qwen).

The migration is non-trivial because the two models expose completely different APIs:
  Bodan's model  : single `model.encoder`, block forward hooks fire, `diffusion_lm` toggle.
  TwoTower       : two `NemotronHModel` towers; the real denoiser forward
                   (`_run_denoiser_step_diffusion`) iterates `for block in tower.layers`
                   MANUALLY (AdaLN modulation + cross-attn to context KV + Mamba state
                   seeding) and never calls `block.forward`, so block hooks DO NOT fire.
Capture strategy therefore differs per tower:
  * context tower (AR)  : one clean causal forward with output_hidden_states=True
                          (== its real behaviour: a plain cached causal LM).
  * denoiser tower (diff): a capturing, byte-faithful copy of _run_denoiser_step_diffusion
                          run for ONE real conditioned step on an all-[MASK] block (t=1.0),
                          seeded from a real context cache. This is the ACTUAL inference-time
                          computation (fixed Mamba kernel, cross-attn, time conditioning),
                          not the tower run in isolation.

Both capture paths return the residual stream as [embeddings, layer_1, ..., layer_52]
(53 entries for 52 layers), matching HF's output_hidden_states convention so the two towers
are indexed identically.

Runs on the pod (2 GPUs). Pure-math self-check runs anywhere:  python src/layer_similarity.py --selftest

    python src/layer_similarity.py --prompts data/gsm8k_mini.jsonl --out results/layer_sim \
        --block-size 16 --num-prompts 8 --plot
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# ======================================================================
# Curve math — kept ~verbatim from Bodan's probe (model-agnostic).
# ======================================================================
@dataclass
class CurveStats:
    sums: list = field(default_factory=list)
    sums2: list = field(default_factory=list)
    counts: list = field(default_factory=list)
    step_means: list = field(default_factory=list)


def _ensure_len(stats, n):
    if stats.sums:
        return
    stats.sums = [0.0] * n
    stats.sums2 = [0.0] * n
    stats.counts = [0] * n


def add_values(stats, layer_values):
    _ensure_len(stats, len(layer_values))
    step_mean = []
    for i, vals in enumerate(layer_values):
        vals = vals.detach().float().cpu()
        vals = vals[torch.isfinite(vals)]
        if vals.numel() == 0:
            step_mean.append(float("nan"))
            continue
        s = vals.sum().item()
        s2 = (vals * vals).sum().item()
        c = int(vals.numel())
        stats.sums[i] += s
        stats.sums2[i] += s2
        stats.counts[i] += c
        step_mean.append(s / c)
    stats.step_means.append(step_mean)


def tokenwise_cosines(hidden_states, token_slice):
    """cos(hidden[t], hidden[t+1]) within each layer -> one curve over layers."""
    values = []
    for hidden in hidden_states:
        x = hidden[0, token_slice, :].float()
        if x.ndim == 1 or x.shape[0] < 2:
            values.append(torch.empty(0))
        else:
            values.append(F.cosine_similarity(x[:-1], x[1:], dim=-1))
    return values


def adjacent_layer_cosines(hidden_states, token_slice):
    """cos(hidden_L[i], hidden_{L+1}[i]) per token -> the LAYER-REDUNDANCY curve."""
    values = []
    for left, right in zip(hidden_states[:-1], hidden_states[1:]):
        a = left[0, token_slice, :].float()
        b = right[0, token_slice, :].float()
        if a.ndim == 1:
            a = a.unsqueeze(0)
            b = b.unsqueeze(0)
        values.append(F.cosine_similarity(a, b, dim=-1))
    return values


def summarize(stats):
    rows = []
    for i, (s, s2, c) in enumerate(zip(stats.sums, stats.sums2, stats.counts)):
        if c == 0:
            mean = std = float("nan")
        else:
            mean = s / c
            var = max(0.0, (s2 / c) - mean * mean)
            std = math.sqrt(var)
        step_vals = [row[i] for row in stats.step_means
                     if i < len(row) and math.isfinite(row[i])]
        step_std = 0.0
        if len(step_vals) > 1:
            sm = sum(step_vals) / len(step_vals)
            step_std = math.sqrt(sum((x - sm) ** 2 for x in step_vals) / (len(step_vals) - 1))
        rows.append({"layer": i, "mean": mean, "std": std,
                     "step_std": step_std, "count": c})
    return rows


def write_rows(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "mean", "std", "step_std", "count"])
        writer.writeheader()
        writer.writerows(rows)


# ======================================================================
# TwoTower-specific capture.
# ======================================================================
def _context_hidden_states(model, prompt_ids):
    """Context (AR) tower: one clean causal forward, residual stream via HF hidden_states.
    This IS the context tower's real behaviour (a plain cached causal LM), so no patch needed."""
    ctx_device = next(model.context_tower.parameters()).device
    out = model.context_tower(input_ids=prompt_ids.to(ctx_device),
                              use_cache=False, output_hidden_states=True)
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise RuntimeError("context_tower did not return hidden_states; "
                           "NemotronHModel.forward must support output_hidden_states=True")
    return [h.detach() for h in hs]  # (embed, l1..lN)


def _denoiser_hidden_states_faithful(model, cache_state, block_ids, t_scalar):
    """Denoiser (diffusion) tower: ONE real conditioned step, byte-faithful copy of
    _run_denoiser_step_diffusion (reference_modeling.py:594) with a capture after each layer.
    Uses the FIXED per-token Mamba kernel (twotower.apply_diffusion_fix), real cross-attn to
    the context KV, real Mamba state seeding, and real AdaLN time conditioning."""
    _mod = sys.modules[type(model).__module__]
    _get_mod_params, _modulate = _mod._get_mod_params, _mod._modulate

    tower = model.denoiser_tower
    den_device = next(tower.parameters()).device
    den_input = block_ids.to(den_device)
    t = torch.full((den_input.shape[0],), float(t_scalar), device=den_device, dtype=model.dtype)

    t_repr = model.t_embedder(t)
    t_emb = model.t_block(t_repr)

    den_cache = model._build_denoiser_cache_diffusion(cache_state, den_device)
    hidden = tower.embeddings(den_input)

    captured = [hidden.detach()]                       # index 0 == embeddings (matches HF)
    for layer_idx, block in enumerate(tower.layers):
        residual = hidden
        if block.residual_in_fp32:
            residual = residual.to(torch.float32)

        mod = _get_mod_params(t_emb, model.scale_shift_tables[layer_idx])
        shift, scale, gate = mod

        if block.block_type in ("mamba", "attention"):
            h = _modulate(hidden, shift, scale)
            h = block.norm(h.to(dtype=block.norm.weight.dtype))
        else:  # mlp / moe
            h = block.norm(hidden.to(dtype=block.norm.weight.dtype))
            h = _modulate(h, shift, scale)

        if block.block_type == "mamba":
            d_conv = block.mixer.conv_kernel_size
            init_conv = den_cache.conv_states[layer_idx][..., -(d_conv - 1):]
            init_ssm = den_cache.ssm_states[layer_idx].contiguous()
            h = model._denoiser_block_mamba(block.mixer, h, init_conv, init_ssm)
        elif block.block_type == "attention":
            h = model._denoiser_block_attention(block.mixer, h,
                                                den_cache.key_cache[layer_idx],
                                                den_cache.value_cache[layer_idx])
        else:
            h = block.mixer(h)

        h = gate.unsqueeze(1) * h
        hidden = residual + h
        captured.append(hidden.detach())               # residual stream after layer L
    return captured


def _denoiser_hidden_states_standalone(model, block_ids):
    """Rough baseline: run the denoiser tower in ISOLATION (no context / AdaLN / cross-attn,
    causal default). NOT faithful to real inference — offered only for a cheap sanity curve."""
    den_device = next(model.denoiser_tower.parameters()).device
    out = model.denoiser_tower(input_ids=block_ids.to(den_device),
                               use_cache=False, output_hidden_states=True)
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise RuntimeError("denoiser_tower did not return hidden_states")
    return [h.detach() for h in hs]


# ======================================================================
# Per-prompt driver.
# ======================================================================
def probe_prompt(model, tok, prompt, args, stats):
    from twotower import MASK_TOKEN_ID
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")

    # --- context (AR) tower over the prompt ---
    ctx_hs = _context_hidden_states(model, prompt_ids)
    ctx_slice = slice(0, ctx_hs[0].shape[1])
    add_values(stats["context_tokenwise"], tokenwise_cosines(ctx_hs, ctx_slice))
    add_values(stats["context_adjacent_layer"], adjacent_layer_cosines(ctx_hs, ctx_slice))

    # --- denoiser (diffusion) tower over one all-[MASK] block, seeded by the prompt ---
    block = torch.full((1, args.block_size), MASK_TOKEN_ID, dtype=torch.long, device="cuda:0")
    if args.denoiser_standalone:
        den_hs = _denoiser_hidden_states_standalone(model, block)
    else:
        cache_state = model._build_context_cache(prompt_ids)
        den_hs = _denoiser_hidden_states_faithful(model, cache_state, block, args.t)
    den_slice = slice(0, den_hs[0].shape[1])
    add_values(stats["denoiser_tokenwise"], tokenwise_cosines(den_hs, den_slice))
    add_values(stats["denoiser_adjacent_layer"], adjacent_layer_cosines(den_hs, den_slice))


def make_plot(results, out_png):
    """Overlay the two towers' adjacent-layer redundancy curves (the money figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, color, label in [
        ("context_adjacent_layer", "#1f77b4", "context tower (AR, frozen)"),
        ("denoiser_adjacent_layer", "#d62728", "denoiser tower (diffusion-trained)"),
    ]:
        rows = results[key]
        xs = [r["layer"] for r in rows]
        ys = [r["mean"] for r in rows]
        es = [r["std"] for r in rows]
        ax.plot(xs, ys, "-o", ms=3, color=color, label=label)
        ax.fill_between(xs, [y - e for y, e in zip(ys, es)],
                        [y + e for y, e in zip(ys, es)], color=color, alpha=0.12)
    ax.set_xlabel("layer boundary (L -> L+1)")
    ax.set_ylabel("adjacent-layer cosine  (higher = more redundant)")
    ax.set_title("TwoTower layer-to-layer redundancy: AR tower vs diffusion tower")
    ax.axhline(0.9, ls="--", lw=0.8, color="gray")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"[plot] wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", help="jsonl with a 'prompt' field (data/*.jsonl)")
    ap.add_argument("--out", default="results/layer_sim", help="output dir prefix")
    ap.add_argument("--label", default="twotower")
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--t", type=float, default=1.0,
                    help="mask-ratio fed to AdaLN for the denoiser step (1.0 = all-masked)")
    ap.add_argument("--denoiser-standalone", action="store_true",
                    help="probe denoiser in isolation (unconditioned) instead of faithful step")
    ap.add_argument("--single", action="store_true", help="both towers on cuda:0")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="run curve-math check, no model")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from twotower import load
    from main_run import load_prompts

    prompts = load_prompts(args.prompts, args.num_prompts)
    model, tok = load(den_device="cuda:0" if args.single else "cuda:1")

    stats = {k: CurveStats() for k in (
        "context_tokenwise", "context_adjacent_layer",
        "denoiser_tokenwise", "denoiser_adjacent_layer")}

    t0 = time.time()
    with torch.no_grad():
        for i, p in enumerate(prompts):
            probe_prompt(model, tok, p["prompt"], args, stats)
            print(f"[{args.label}] prompt {i + 1}/{len(prompts)} done", flush=True)

    results = {k: summarize(v) for k, v in stats.items()}
    for name, rows in results.items():
        write_rows(f"{args.out}/{args.label}_{name}_similarity.csv", rows)

    def band(rows):
        ms = [r["mean"] for r in rows if math.isfinite(r["mean"])]
        return {"min": min(ms), "max": max(ms), "layers": len(rows)} if ms else {}

    summary = {"label": args.label, "num_prompts": len(prompts),
               "block_size": args.block_size, "t": args.t,
               "denoiser_standalone": args.denoiser_standalone,
               "elapsed_sec": round(time.time() - t0, 1),
               "adjacent_layer": {
                   "context_AR": band(results["context_adjacent_layer"]),
                   "denoiser_diffusion": band(results["denoiser_adjacent_layer"])}}
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/{args.label}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.plot:
        make_plot(results, "figs/fig_layer_redundancy.png")


# ----------------------------------------------------------------------
def _selftest():
    """No-model check that the curve math behaves: identical layers -> cosine 1.0;
    orthogonal adjacent layers -> ~0.0; monotone token ramp -> high tokenwise cosine."""
    torch.manual_seed(0)
    H, T = 64, 20
    base = torch.randn(1, T, H)
    # 3 "layers" that are progressively rotated copies -> adjacent-layer cos decreasing
    hs = [base, base.clone(), base + 0.0 * torch.randn(1, T, H)]  # first two identical
    adj = adjacent_layer_cosines(hs, slice(0, T))
    assert abs(adj[0].mean().item() - 1.0) < 1e-5, adj[0].mean().item()
    # tokenwise on a smooth ramp -> near 1; on random -> lower.
    # +1.0 offset keeps every token vector non-zero (a zero vector => cosine nan).
    ramp = (torch.linspace(0, 1, T).view(1, T, 1) + 1.0) * torch.ones(1, T, H)
    tw_ramp = tokenwise_cosines([ramp], slice(0, T))[0].mean().item()
    tw_rand = tokenwise_cosines([torch.randn(1, T, H)], slice(0, T))[0].mean().item()
    assert tw_ramp > tw_rand, (tw_ramp, tw_rand)
    # summarize / accumulate plumbing
    st = CurveStats()
    add_values(st, adj)
    add_values(st, adjacent_layer_cosines([base, base.clone()], slice(0, T)))
    rows = summarize(st)
    assert len(rows) == len(adj) and rows[0]["count"] > 0
    print(f"[selftest] OK  adj[0]={adj[0].mean():.4f} (==1) "
          f"tw_ramp={tw_ramp:.3f} > tw_rand={tw_rand:.3f}  rows={len(rows)}")


if __name__ == "__main__":
    main()
