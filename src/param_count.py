"""Per-module parameter breakdown from config.json ARITHMETIC ONLY.

Runs anywhere (your laptop, no GPU, no torch, no mamba-ssm) because it never imports
the model — it derives every matrix shape from config.json the way the modeling code
builds them. This reproduces the 30B-A3B accounting:
  ~30B total / ~3B activated PER TOWER, ~63B across both towers.

NOTE: this counts ONE NemotronH backbone tower (modeling_nemotron_h.py). The denoiser
tower additionally has per-layer cross-attention + AdaLN time conditioning defined in
modeling_nemotron_twotower.py (not yet read) — those are flagged as TODO extras.
Pass --config to point at the model's config.json; otherwise the embedded one is used.
"""
import argparse
import json

# config.json as provided by the user (embedded default)
CFG = {
    "hidden_size": 2688, "num_hidden_layers": 52,
    "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
    "vocab_size": 131072, "tie_word_embeddings": False,
    # mamba
    "mamba_num_heads": 64, "mamba_head_dim": 64, "n_groups": 8,
    "ssm_state_size": 128, "conv_kernel": 4, "use_conv_bias": True, "use_bias": False,
    # attention
    "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
    "attention_bias": False,
    # moe
    "n_routed_experts": 128, "num_experts_per_tok": 6, "moe_intermediate_size": 1856,
    "moe_shared_expert_intermediate_size": 3712, "mlp_bias": False,
}


def module_params(c):
    H = c["hidden_size"]

    # --- Mamba-2 mixer (in_proj, conv1d, dt/A/D, gated norm, out_proj) ---
    mi = c["mamba_num_heads"] * c["mamba_head_dim"]                 # 4096
    conv_dim = mi + 2 * c["n_groups"] * c["ssm_state_size"]        # 6144
    proj = mi + conv_dim + c["mamba_num_heads"]                    # 10304
    mamba = (
        H * proj + mi * H                                          # in_proj + out_proj
        + conv_dim * c["conv_kernel"] + (conv_dim if c["use_conv_bias"] else 0)  # conv1d
        + 3 * c["mamba_num_heads"]                                 # dt_bias + A_log + D
        + mi                                                       # gated RMSNorm weight
    )

    # --- Attention (GQA: 32 q heads, 2 kv heads, head_dim 128) ---
    hd, nh, nkv = c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
    attn = H * (nh * hd) + 2 * H * (nkv * hd) + (nh * hd) * H       # q + k + v + o

    # --- MoE (128 experts of relu^2 MLP up/down, 1 shared, fp32 router) ---
    expert = 2 * H * c["moe_intermediate_size"]
    shared = 2 * H * c["moe_shared_expert_intermediate_size"]
    router = c["n_routed_experts"] * H
    moe_total = c["n_routed_experts"] * expert + shared + router
    moe_active = c["num_experts_per_tok"] * expert + shared + router

    return dict(mamba=mamba, attn=attn, moe_total=moe_total, moe_active=moe_active,
                expert=expert, shared=shared, block_norm=H)


def count_tower(c):
    m = module_params(c)
    p = c["hybrid_override_pattern"]
    nM, nE, nA = p.count("M"), p.count("E"), p.count("*")
    L = c["num_hidden_layers"]
    embed = c["vocab_size"] * c["hidden_size"]
    lm_head = 0 if c["tie_word_embeddings"] else c["vocab_size"] * c["hidden_size"]
    norms = L * m["block_norm"] + c["hidden_size"]  # per-block + final

    rows = [
        (f"mamba2   x{nM}", nM * m["mamba"]),
        (f"moe      x{nE} (total)", nE * m["moe_total"]),
        (f"attention x{nA}", nA * m["attn"]),
        ("embed + lm_head", embed + lm_head),
        ("norms", norms),
    ]
    total = sum(v for _, v in rows)
    active = (nM * m["mamba"] + nE * m["moe_active"] + nA * m["attn"]
              + embed + lm_head + norms)
    return rows, total, active, (nM, nE, nA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="path to config.json (defaults to embedded)")
    args = ap.parse_args()
    c = json.load(open(args.config)) if args.config else CFG

    rows, total, active, (nM, nE, nA) = count_tower(c)
    print(f"\nPattern: {nM} Mamba-2 / {nE} MoE / {nA} attention  (per tower, 52 layers)\n")
    print(f"{'module':<26}{'params':>14}")
    print("-" * 40)
    for name, v in rows:
        print(f"{name:<26}{v/1e9:>12.3f}B")
    print("-" * 40)
    print(f"{'TOWER total':<26}{total/1e9:>12.3f}B")
    print(f"{'TOWER activated':<26}{active/1e9:>12.3f}B")
    print(f"\nBoth towers (x2, backbone only): total {2*total/1e9:.1f}B, "
          f"activated {2*active/1e9:.2f}B")
    print("[TODO] denoiser adds per-layer cross-attn + AdaLN "
          "(needs modeling_nemotron_twotower.py) -> Analysis #3")


if __name__ == "__main__":
    main()
