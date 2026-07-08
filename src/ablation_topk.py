"""Ablation 4: change the number of active MoE experts per token at INFERENCE (top-k).

Corresponds to the paper's MoE (128 routed / 6 active + 1 shared). The paper never probes
the inference-time quality-vs-compute tradeoff of top-k; this does.

IMPORTANT (verified against modeling_nemotron_h.py): NemotronHTopkRouter caches
`self.top_k = config.num_experts_per_tok` at __init__ and get_topk_indices() uses
`self.top_k`. So editing config after load does NOTHING — we must set `.top_k` on every
router module. This script does that (switchable via --topk) and VERIFIES the change
actually altered the number of selected experts via a forward hook.

    python src/ablation_topk.py --prompts data/gsm8k_mini.jsonl --out results/abl_topk.pkl
"""
import argparse
import os
import pickle
import types

import torch

from twotower import load, MASK_TOKEN_ID
from main_run import run_prompt, load_prompts   # reuse the exact Part-1 logging


def find_routers(model, denoiser_only=True):
    routers = []
    for name, m in model.named_modules():
        if m.__class__.__name__ == "NemotronHTopkRouter":
            if denoiser_only and not any(k in name.lower() for k in ("denois", "diffusion")):
                continue
            routers.append((name, m))
    return routers


def set_topk(model, k, scope):
    routers = find_routers(model, denoiser_only=(scope == "denoiser"))
    if not routers and scope == "denoiser":
        print("[topk] no denoiser-named routers found -> falling back to ALL routers")
        routers = find_routers(model, denoiser_only=False)
    old = sorted({int(m.top_k) for _, m in routers})
    for _, m in routers:
        m.top_k = k
    for _, m in routers:                       # verification: attribute really changed
        assert m.top_k == k, "failed to set top_k"
    print(f"[topk] set top_k={k} on {len(routers)} routers (was {old})")
    return len(routers)


def verify_active_count(model, tok, k):
    """Prove the change bites: hook a router and read the width of its topk_indices."""
    cap = {}

    def hook(m, i, o):
        idx = o[0] if isinstance(o, (tuple, list)) else o
        cap["k"] = int(idx.shape[-1])

    handles = [m.register_forward_hook(hook) for _, m in find_routers(model, False)[:1]]
    ids = tok("verify", return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        model.generate_mask_diffusion(ids, max_new_tokens=16, block_size=16,
                                       steps_per_block=2, mask_token_id=MASK_TOKEN_ID,
                                       temperature=0.0, confidence_threshold=0.8)
    for h in handles:
        h.remove()
    got = cap.get("k")
    print(f"[verify] router selected {got} experts/token (expected {k})")
    assert got == k, "top_k change did NOT take effect in routing!"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="results/abl_topk.pkl")
    ap.add_argument("--topk", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--scope", choices=["denoiser", "all"], default="denoiser")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--single", action="store_true")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts, args.limit)
    model, tok = load(den_device="cuda:0" if args.single else "cuda:1")

    gen_args = types.SimpleNamespace(max_new=args.max_new, block_size=args.block_size,
                                     steps=args.steps, gamma=args.gamma,
                                     temperature=args.temperature)

    results = []
    for k in args.topk:
        print(f"\n===== top_k = {k} ({args.scope}) =====")
        n = set_topk(model, k, args.scope)
        verify_active_count(model, tok, k)
        for p in prompts:
            summary, _ = run_prompt(model, tok, p["prompt"], gen_args)
            results.append(dict(topk=k, scope=args.scope, n_routers=n,
                                prompt_id=p["id"], summary=summary,
                                reference=p.get("reference")))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dict(config=vars(args), results=results), f)
    print(f"\nLOGGING OK -> {args.out}  ({len(results)} runs across top_k={args.topk})")


if __name__ == "__main__":
    main()
