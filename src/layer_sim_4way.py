"""4-way controlled layer probe: disentangle WHY the denoiser tower's token
representations collapse (seen in layer_token_structure.py).

The collapse in layer_token_structure.py could come from FOUR confounded causes:
  (a) weight drift     — the ~8% mask-diffusion training reshaped the representation dynamics
  (b) [MASK] input     — the real denoiser input is all-[MASK] (identical embeddings -> trivially similar)
  (c) bidirectional attn — the denoiser's attention layers run is_causal=False + concat context KV,
                           one global bidirectional mix can pull every token the same way
  (d) seeded-state scale — the seeded Mamba SSM state is O(1e3)+ on long context; if it dwarfs the
                           normal hidden magnitude it dominates the residual stream -> uniform tokens

This probe runs the SAME clean text prompt through FOUR paths that turn these factors on one at a
time, so the curves localize the cause:

  1. ctx_clean  : context_tower,  clean text, causal,  no t, no seeding      (baseline, = real AR)
  2. den_clean  : denoiser_tower, clean text, causal,  no t, no seeding      (★ isolates WEIGHT DRIFT)
  3. den_bidir  : denoiser_tower, clean text, BIDIR,   no t, no seeding      (isolates bidirectional attn)
  4. den_natural: denoiser_tower, all-[MASK] block, bidir + seeded + t       (the real diffusion path)

Decision tree (printed at the end):
  den_clean ~= ctx_clean                 -> NOT weight drift; collapse is input/attn -> (a) rejected
  den_clean still collapses at L6        -> real WEIGHT DRIFT (a) holds
  den_clean normal but den_bidir collapse-> bidirectional attention (c)
  den_bidir normal but den_natural coll. -> [MASK] input and/or seeded state (b)/(d); the SSM-magnitude
                                            diagnostic (init_ssm vs hidden) then separates (d).

Metrics per path:
  * token_collapse[L] = mean_{i!=j} cos(h_L[i], h_L[j])   (global token collapse; L over 53 hidden states)
  * adjacent_token[L] = mean_{|i-j|=1} cos                (local smoothness)
  * rel_update[L]     = mean_t ||mixer_out_L[t]|| / ||residual_in_L[t]||   (how much layer L injects;
                        NOT washed out by the residual add, unlike adjacent-LAYER cosine)

Captures are INLINE (not hooks): the tower's real forward paths (_forward_tower_with_cache /
_run_denoiser_step_diffusion) iterate block.norm/block.mixer manually and NEVER call block.forward,
so nn.Module forward hooks on NemotronHBlock do not fire. Every capture is .detach().float().cpu()
immediately (52 fp32 layers would otherwise blow up memory).

    python src/layer_sim_4way.py --selftest
    python src/layer_sim_4way.py --out results/layer_sim_4way.npz --plot           # builtin 5 prompts
    python src/layer_sim_4way.py --prompts data/gsm8k_mini.jsonl --num-prompts 5 --plot
"""
import argparse
import inspect
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


ATTN_LAYERS = [5, 12, 19, 26, 33, 42]  # '*' positions in MEMEM*E... (52 layers)

# 5 diverse prompt types (math / code / prose / fact / long), ~50-80 tokens each.
BUILTIN_PROMPTS = [
    ("math", "Natalia sold clips to 48 of her friends in April, and then she sold half as "
              "many clips in May. How many clips did she sell altogether in April and May? "
              "Show your reasoning step by step before giving the final number."),
    ("code", "def merge_sorted(a, b):\n    \"\"\"Merge two ascending lists into one sorted "
              "list without using sorted(). Return the merged list.\"\"\"\n    i = j = 0\n    "
              "out = []\n    while i < len(a) and j < len(b):\n"),
    ("prose", "The old lighthouse stood alone on the granite cliff, its slow beam sweeping "
               "across the black water while the storm gathered force over the restless sea, "
               "and the keeper wondered whether the ships had already turned back to harbour."),
    ("fact", "The capital of France is Paris, a city on the river Seine that has been the "
              "political and cultural centre of the country for centuries and is known "
              "worldwide for the Eiffel Tower, the Louvre museum, and Notre-Dame cathedral."),
    ("long", "In distributed systems a consensus protocol lets a set of unreliable machines "
              "agree on a single value even when some of them crash or messages are delayed; "
              "Paxos and Raft are the two classic algorithms, and Raft is usually preferred "
              "because its leader election and log replication are easier to reason about."),
]


# ======================================================================
# Token-pair structure math.
# ======================================================================
def cosine_matrix(x_TD):
    x = x_TD.float()
    if x.ndim == 1:
        x = x.unsqueeze(0)
    xn = F.normalize(x, dim=-1)
    return xn @ xn.T


def offdiag_mean(S):
    T = S.shape[0]
    if T < 2:
        return float("nan")
    return ((S.sum() - torch.diagonal(S).sum()) / (T * T - T)).item()


def diag_offset_mean(S, d=1):
    T = S.shape[0]
    if T <= d:
        return float("nan")
    return torch.diagonal(S, d).mean().item()


def structure_of(hidden_list):
    """hidden_list: [ (T,D) cpu float ] x (num hidden states). Returns collapse+adjacent curves."""
    collapse, adjacent = [], []
    for h in hidden_list:
        S = cosine_matrix(h)
        collapse.append(offdiag_mean(S))
        adjacent.append(diag_offset_mean(S, 1))
    return collapse, adjacent


def rel_update_of(mixer_outs, residual_ins):
    """per-layer mean_t ||mixer_out[t]|| / ||residual_in[t]||."""
    out = []
    for mo, ri in zip(mixer_outs, residual_ins):
        num = mo.float().norm(dim=-1)
        den = ri.float().norm(dim=-1).clamp(min=1e-6)
        out.append((num / den).mean().item())
    return out


# ======================================================================
# Unified capture for the three CLEAN-TEXT paths (ctx_clean/den_clean/den_bidir).
# Mirrors NemotronHTwoTowerForCausalLM._forward_tower_with_cache (reference_modeling.py:254),
# swapping attention causal<->bidirectional; no time conditioning; fresh cache = no seeding.
# ======================================================================
def _bidir_attention(mixer, hidden):
    """Bidirectional self-attention over just the block (no context KV, is_causal=False).
    Same projections as _denoiser_block_attention but without the concatenated context cache."""
    from twotower import MASK_TOKEN_ID  # noqa: F401  (keep import local & lazy)
    B, Lq, _ = hidden.shape
    q = mixer.q_proj(hidden).view(B, Lq, mixer.num_heads, mixer.head_dim).transpose(1, 2)
    k = mixer.k_proj(hidden).view(B, Lq, mixer.num_key_value_heads, mixer.head_dim).transpose(1, 2)
    v = mixer.v_proj(hidden).view(B, Lq, mixer.num_key_value_heads, mixer.head_dim).transpose(1, 2)
    from reference_modeling import repeat_kv
    k = repeat_kv(k, mixer.num_key_value_groups)
    v = repeat_kv(v, mixer.num_key_value_groups)
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    a = a.transpose(1, 2).contiguous().view(B, Lq, mixer.num_heads * mixer.head_dim)
    return mixer.o_proj(a)


def capture_clean(model, tower, input_ids, bidirectional):
    """Return (hidden_states[53], mixer_outs[52], residual_ins[52]) all cpu float, one clean pass."""
    device = next(tower.parameters()).device
    ids = input_ids.to(device)
    cache = model._make_cache(model.config, ids.shape[0], model.dtype, device)
    cache_position = torch.arange(ids.shape[1], device=device)
    hidden = tower.embeddings(ids)
    causal_mask = tower._update_causal_mask(None, hidden, cache_position)

    hs = [hidden[0].detach().float().cpu()]
    mixer_outs, residual_ins = [], []
    for layer_idx, block in enumerate(tower.layers):
        residual = hidden
        if block.residual_in_fp32:
            residual = residual.to(torch.float32)
        h = block.norm(hidden.to(dtype=block.norm.weight.dtype))

        if block.block_type == "mamba":
            h = block.mixer(h, cache_params=cache, cache_position=cache_position)
        elif block.block_type == "attention":
            if bidirectional:
                h = _bidir_attention(block.mixer, h)
            else:
                h, _, _ = block.mixer(h, attention_mask=causal_mask,
                                      past_key_value=cache, cache_position=cache_position)
        elif block.block_type in ("mlp", "moe"):
            h = block.mixer(h)
        else:
            raise ValueError(block.block_type)

        mixer_outs.append(h[0].detach().float().cpu())
        residual_ins.append(residual[0].detach().float().cpu())
        hidden = residual + h
        hs.append(hidden[0].detach().float().cpu())
    return hs, mixer_outs, residual_ins


# ======================================================================
# den_natural: real diffusion path (all-[MASK] block + seeded cache + t) with capture
# + SSM seeding-magnitude diagnostic. Mirrors _run_denoiser_step_diffusion (ref:594).
# ======================================================================
def capture_natural(model, cache_state, block_ids, t_scalar, ssm_diag):
    mod = sys.modules[type(model).__module__]
    _get_mod_params, _modulate = mod._get_mod_params, mod._modulate
    tower = model.denoiser_tower
    den_device = next(tower.parameters()).device
    den_input = block_ids.to(den_device)

    t_emb = None
    if t_scalar is not None:
        t = torch.full((den_input.shape[0],), float(t_scalar), device=den_device, dtype=model.dtype)
        t_emb = model.t_block(model.t_embedder(t))

    den_cache = model._build_denoiser_cache_diffusion(cache_state, den_device)
    hidden = tower.embeddings(den_input)

    hs = [hidden[0].detach().float().cpu()]
    mixer_outs, residual_ins = [], []
    for layer_idx, block in enumerate(tower.layers):
        residual = hidden
        if block.residual_in_fp32:
            residual = residual.to(torch.float32)
        mod_p = None
        if t_emb is not None:
            mod_p = _get_mod_params(t_emb, model.scale_shift_tables[layer_idx])
            shift, scale, gate = mod_p

        if block.block_type in ("mamba", "attention"):
            h = hidden
            if mod_p is not None:
                h = _modulate(h, shift, scale)
            h = block.norm(h.to(dtype=block.norm.weight.dtype))
        else:
            h = block.norm(hidden.to(dtype=block.norm.weight.dtype))
            if mod_p is not None:
                h = _modulate(h, shift, scale)

        if block.block_type == "mamba":
            d_conv = block.mixer.conv_kernel_size
            init_conv = den_cache.conv_states[layer_idx][..., -(d_conv - 1):]
            init_ssm = den_cache.ssm_states[layer_idx].contiguous()
            # --- (d) diagnostic: seeded SSM state magnitude vs hidden magnitude ---
            ssm_diag.append({
                "layer": layer_idx,
                "init_ssm_max": init_ssm.abs().max().item(),
                "init_ssm_mean": init_ssm.abs().mean().item(),
                "hidden_max": h.abs().max().item(),
                "hidden_mean": h.abs().mean().item(),
            })
            h = model._denoiser_block_mamba(block.mixer, h, init_conv, init_ssm)
        elif block.block_type == "attention":
            h = model._denoiser_block_attention(block.mixer, h,
                                                den_cache.key_cache[layer_idx],
                                                den_cache.value_cache[layer_idx])
        else:
            h = block.mixer(h)

        mixer_outs.append(h[0].detach().float().cpu())
        residual_ins.append(residual[0].detach().float().cpu())
        if mod_p is not None:
            h = gate.unsqueeze(1) * h
        hidden = residual + h
        hs.append(hidden[0].detach().float().cpu())
    return hs, mixer_outs, residual_ins


# ======================================================================
# Driver.
# ======================================================================
PATHS = ["ctx_clean", "den_clean", "den_bidir", "den_natural"]


def probe_all(model, tok, prompt, args, agg, ssm_diag_store, keep_prompt0):
    from twotower import MASK_TOKEN_ID
    prompt_ids = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=args.max_len).input_ids

    caps = {}
    caps["ctx_clean"] = capture_clean(model, model.context_tower, prompt_ids, bidirectional=False)
    caps["den_clean"] = capture_clean(model, model.denoiser_tower, prompt_ids, bidirectional=False)
    caps["den_bidir"] = capture_clean(model, model.denoiser_tower, prompt_ids, bidirectional=True)

    cache_state = model._build_context_cache(prompt_ids.to(next(model.context_tower.parameters()).device))
    nat_len = args.nat_block if args.nat_block > 0 else prompt_ids.shape[1]
    block = torch.full((1, nat_len), MASK_TOKEN_ID, dtype=torch.long)
    ssm_diag = []
    caps["den_natural"] = capture_natural(model, cache_state, block, args.t, ssm_diag)
    if keep_prompt0:
        ssm_diag_store.extend(ssm_diag)

    for name in PATHS:
        hs, mixer_outs, residual_ins = caps[name]
        collapse, adjacent = structure_of(hs)
        rel = rel_update_of(mixer_outs, residual_ins)
        agg[name]["collapse"].append(collapse)
        agg[name]["adjacent"].append(adjacent)
        agg[name]["rel"].append(rel)
        if keep_prompt0:
            agg[name]["heat0"] = cosine_matrix(hs[args.heat_layer]).numpy()


def _mean_rows(rows):
    """mean over prompts of a list of equal-length per-layer lists (nan-safe)."""
    a = np.array(rows, dtype=float)
    return np.nanmean(a, axis=0)


def make_plot(curves, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"ctx_clean": ("#1f77b4", "ctx_clean (AR baseline)"),
             "den_clean": ("#d62728", "den_clean (causal, no seed/t → weight drift)"),
             "den_bidir": ("#ff7f0e", "den_bidir (bidirectional attn)"),
             "den_natural": ("#2ca02c", "den_natural (MASK+seed+t, real)")}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for name in PATHS:
        c, lbl = style[name]
        col = curves[name]["collapse"]; adj = curves[name]["adjacent"]; rel = curves[name]["rel"]
        axes[0].plot(range(len(col)), col, "-o", ms=2.5, color=c, label=lbl)
        axes[1].plot(range(len(adj)), adj, "-o", ms=2.5, color=c, label=lbl)
        axes[2].plot(range(len(rel)), rel, "-o", ms=2.5, color=c, label=lbl)
    for ax in axes:
        for L in ATTN_LAYERS:
            ax.axvline(L, ls=":", lw=0.7, color="gray")
        ax.set_xlabel("layer  (dotted = attention layers 5/12/19/26/33/42)")
    axes[0].set_title("token collapse (mean off-diag cosine)"); axes[0].set_ylabel("cosine")
    axes[1].set_title("local smoothness (adjacent-token cosine)")
    axes[2].set_title("rel_update  ||mixer_out|| / ||residual_in||"); axes[2].set_ylabel("ratio")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("4-way controlled layer probe: what causes the denoiser token collapse?")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"[plot] wrote {out_png}")


def print_verdict(curves):
    """Plateau = mean collapse over layers 8..40 (post-first-attention, pre-endpoint)."""
    def plateau(name):
        c = np.asarray(curves[name]["collapse"], dtype=float)
        return float(np.nanmean(c[8:41]))
    p = {n: plateau(n) for n in PATHS}
    print("\n=== VERDICT (token-collapse plateau, layers 8..40; higher = more collapsed) ===")
    for n in PATHS:
        print(f"  {n:12s} plateau = {p[n]:.3f}")
    gap_weight = p["den_clean"] - p["ctx_clean"]
    gap_bidir = p["den_bidir"] - p["den_clean"]
    gap_natural = p["den_natural"] - p["den_bidir"]
    print(f"\n  den_clean - ctx_clean  = {gap_weight:+.3f}   (weight drift (a))")
    print(f"  den_bidir - den_clean  = {gap_bidir:+.3f}   (bidirectional attn (c))")
    print(f"  den_natural- den_bidir = {gap_natural:+.3f}   (MASK input / seeding (b)/(d))")
    lead = max([("a", gap_weight), ("c", gap_bidir), ("bd", gap_natural)], key=lambda kv: kv[1])
    print(f"\n  >>> largest single-factor jump: ({lead[0]})  Δ={lead[1]:+.3f}")
    if gap_weight > 0.15:
        print("  >>> den_clean already collapses on clean causal text -> WEIGHT DRIFT (a) is real.")
    else:
        print("  >>> den_clean ~ ctx_clean -> NOT weight drift; look at (c)/(b)/(d).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", help="jsonl with 'prompt' field (default: builtin 5 diverse)")
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--out", default="results/layer_sim_4way.npz")
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--nat-block", type=int, default=0,
                    help="den_natural MASK block length; 0 = match prompt length")
    ap.add_argument("--t", type=float, default=1.0, help="mask-ratio fed to AdaLN for den_natural")
    ap.add_argument("--heat-layer", type=int, default=26)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from twotower import load
    from main_run import load_prompts

    if args.prompts:
        prompts = [p["prompt"] for p in load_prompts(args.prompts, args.num_prompts)]
    else:
        prompts = [t for _, t in BUILTIN_PROMPTS][:args.num_prompts]

    model, tok = load(den_device="cuda:0" if args.single else "cuda:1")
    print(">>> _forward_tower_with_cache signature:",
          inspect.signature(model._forward_tower_with_cache))
    print(f">>> attention layers (bidirectional in denoiser): {ATTN_LAYERS}")

    agg = {n: {"collapse": [], "adjacent": [], "rel": [], "heat0": None} for n in PATHS}
    ssm_diag_store = []
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            probe_all(model, tok, prompt, args, agg, ssm_diag_store, keep_prompt0=(i == 0))
            print(f"[4way] prompt {i + 1}/{len(prompts)} done", flush=True)

    curves = {n: {"collapse": _mean_rows(agg[n]["collapse"]),
                  "adjacent": _mean_rows(agg[n]["adjacent"]),
                  "rel": _mean_rows(agg[n]["rel"])} for n in PATHS}

    # --- SSM seeding-magnitude diagnostic (test (d)) ---
    print("\n=== den_natural seeding diagnostic (init_ssm vs hidden, prompt 0) ===")
    ratios = []
    for d in ssm_diag_store:
        r = d["init_ssm_max"] / max(1e-9, d["hidden_max"])
        ratios.append(r)
        if d["layer"] in (0, 2, 4, 20, 40, 50):
            print(f"  L{d['layer']:2d}  init_ssm(max={d['init_ssm_max']:.1f} mean={d['init_ssm_mean']:.3f})  "
                  f"hidden(max={d['hidden_max']:.2f} mean={d['hidden_mean']:.3f})  ratio={r:.1f}x")
    if ratios:
        print(f"  seeded-state/hidden max-ratio: median={np.median(ratios):.1f}x  max={max(ratios):.1f}x")
        print(f"  >>> if ratio >> 100x, seeded state dominates residual -> (d) is a real driver.")

    print_verdict(curves)

    # --- save npz ---
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save = {}
    for n in PATHS:
        save[f"{n}_collapse"] = curves[n]["collapse"]
        save[f"{n}_adjacent"] = curves[n]["adjacent"]
        save[f"{n}_rel_update"] = curves[n]["rel"]
        if agg[n]["heat0"] is not None:
            save[f"{n}_heat0"] = agg[n]["heat0"]
    save["attn_layers"] = np.array(ATTN_LAYERS)
    save["ssm_diag"] = np.array([[d["layer"], d["init_ssm_max"], d["init_ssm_mean"],
                                  d["hidden_max"], d["hidden_mean"]] for d in ssm_diag_store])
    np.savez(args.out, **save)
    print(f"\n[save] wrote {args.out}  ({len(prompts)} prompts, heat_layer=L{args.heat_layer})")

    if args.plot:
        make_plot(curves, "figs/fig_layer_4way.png")


# ----------------------------------------------------------------------
def _selftest():
    T, D = 8, 16
    same = torch.ones(T, D)
    assert abs(offdiag_mean(cosine_matrix(same)) - 1.0) < 1e-5
    orth = torch.eye(T)
    assert abs(offdiag_mean(cosine_matrix(orth))) < 1e-6
    # rel_update: mixer_out half the norm of residual -> 0.5
    ri = [torch.ones(4, D) * 2.0]
    mo = [torch.ones(4, D) * 1.0]
    r = rel_update_of(mo, ri)[0]
    assert abs(r - 0.5) < 1e-5, r
    # structure_of plumbing
    col, adj = structure_of([same, orth])
    assert abs(col[0] - 1.0) < 1e-5 and abs(col[1]) < 1e-6
    print(f"[selftest] OK  same_offdiag=1.0  orth_offdiag~0  rel_update={r:.3f}(==0.5)  "
          f"attn_layers={ATTN_LAYERS}")


if __name__ == "__main__":
    main()
