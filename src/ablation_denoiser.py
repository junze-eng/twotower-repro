"""Ablations 3 (freeze AdaLN time) and 2 (disable Mamba state seeding), implemented as a
switchable verbatim monkey-patch of NemotronHTwoTowerForCausalLM._run_denoiser_step_diffusion.

Two INDEPENDENT flags, read from model attributes so either can be toggled alone:
  _freeze_time_value : float | None   (ablation 3) — replace the incoming mask-ratio t
                                        with a constant before the adaLN time embedder.
  _disable_mamba_seed: bool           (ablation 2) — feed ZERO initial conv/ssm states to
                                        the denoiser Mamba layers instead of the context
                                        tower's states (i.e. no cross-tower state seeding).

Ablation 2 passes initial_states=None (confirmed against _denoiser_block_mamba: both
causal_conv1d_fn and mamba_chunk_scan_combined natively accept None = "no initial state").
This is the true "no seeding" semantics.

The copy of _run_denoiser_step_diffusion below is byte-identical to the model's, except
the lines marked ABLATION-2 / ABLATION-3. Original method is restored after the run.

    python src/ablation_denoiser.py --prompts data/gsm8k_mini.jsonl --out results/abl_denoiser.jsonl
"""
import argparse
import json
import os
import time

import torch

from twotower import load, MASK_TOKEN_ID, get_nfe
from main_run import load_prompts


def _ablatable_denoiser_step(self, block_ids, cache_state, t=None, den_cache=None):
    # _get_mod_params / _modulate are MODULE-LEVEL helpers in the real modeling file, not
    # methods on self. Resolve them from the model's actual (trust_remote_code) module so
    # this patched copy behaves identically.
    import sys
    _mod = sys.modules[type(self).__module__]
    _get_mod_params = _mod._get_mod_params
    _modulate = _mod._modulate

    ctx_len = cache_state["ctx_len"]
    tower = self.denoiser_tower
    den_device = next(tower.parameters()).device
    den_input = block_ids.to(den_device)
    L = den_input.shape[1]

    # ABLATION-3: freeze the time conditioning to a constant mask-ratio
    freeze_val = getattr(self, "_freeze_time_value", None)
    if freeze_val is not None and t is not None:
        t = torch.full_like(t, float(freeze_val))
        self._t_trace.append(float(t.reshape(-1)[0].detach().cpu()))
    elif t is not None:
        self._t_trace.append(float(t.reshape(-1)[0].detach().cpu()))

    t_emb = None
    if t is not None:
        t_dev = t.to(device=den_device, dtype=self.dtype)
        t_repr = self.t_embedder(t_dev)
        t_emb = self.t_block(t_repr)

    if den_cache is None:
        den_cache = self._build_denoiser_cache_diffusion(cache_state, den_device)

    hidden = tower.embeddings(den_input)

    disable_seed = getattr(self, "_disable_mamba_seed", False)  # ABLATION-2 flag

    for layer_idx, block in enumerate(tower.layers):
        residual = hidden
        if block.residual_in_fp32:
            residual = residual.to(torch.float32)

        mod = None
        if t_emb is not None:
            mod = _get_mod_params(t_emb, self.scale_shift_tables[layer_idx])
            shift, scale, gate = mod

        if block.block_type in ("mamba", "attention"):
            h = hidden
            if mod is not None:
                h = _modulate(h, shift, scale)
            h = block.norm(h.to(dtype=block.norm.weight.dtype))
        else:  # mlp / moe
            h = block.norm(hidden.to(dtype=block.norm.weight.dtype))
            if mod is not None:
                h = _modulate(h, shift, scale)

        if block.block_type == "mamba":
            d_conv = block.mixer.conv_kernel_size
            init_conv = den_cache.conv_states[layer_idx][..., -(d_conv - 1):]
            init_ssm = den_cache.ssm_states[layer_idx].contiguous()
            if disable_seed:                              # ABLATION-2: no cross-tower seed
                # _denoiser_block_mamba natively supports None (causal_conv1d_fn and
                # mamba_chunk_scan_combined both treat initial_states=None as "no state").
                # None is the true "no seeding" semantics, cleaner than zeros.
                init_conv = None
                init_ssm = None
                self._seed_zeroed_layers += 1
            h = self._denoiser_block_mamba(block.mixer, h, init_conv, init_ssm)
        elif block.block_type == "attention":
            ctx_k = den_cache.key_cache[layer_idx]
            ctx_v = den_cache.value_cache[layer_idx]
            h = self._denoiser_block_attention(block.mixer, h, ctx_k, ctx_v)
        elif block.block_type in ["mlp", "moe"]:
            h = block.mixer(h)
        else:
            raise ValueError(f"Unknown block_type: {block.block_type}")

        if mod is not None:
            h = gate.unsqueeze(1) * h

        hidden = residual + h

    hidden = tower.norm_f(hidden)
    logits = self.lm_head(hidden.to(self.lm_head.weight.dtype)).float()
    return logits


def run(model, tok, prompt, args, mode):
    # set flags for this mode (independent)
    model._freeze_time_value = args.freeze_t if mode == "freeze_time" else None
    model._disable_mamba_seed = (mode == "disable_seed")
    model._t_trace = []
    model._seed_zeroed_layers = 0

    input_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")
    plen = input_ids.shape[1]
    t0 = time.time()
    out = model.generate_mask_diffusion(
        input_ids, max_new_tokens=args.max_new, block_size=args.block_size,
        steps_per_block=args.steps, mask_token_id=MASK_TOKEN_ID,
        temperature=args.temperature, confidence_threshold=args.gamma,
        eos_token_id=tok.eos_token_id)
    wall = time.time() - t0
    nfe = get_nfe(model)
    text = tok.decode(out[0][plen:], skip_special_tokens=True)
    tvals = model._t_trace

    # --- per-mode verification ---
    if mode == "freeze_time":
        uniq = sorted(set(round(v, 4) for v in tvals))
        assert uniq == [round(args.freeze_t, 4)], f"t not constant when frozen: {uniq}"
        print(f"    [verify] t held constant at {args.freeze_t} over {len(tvals)} calls")
    elif mode == "disable_seed":
        assert model._seed_zeroed_layers > 0, "no mamba layer had its seed zeroed"
        print(f"    [verify] zeroed init state on {model._seed_zeroed_layers} mamba calls")
    else:
        print(f"    [baseline] t varied over {len(set(round(v,3) for v in tvals))} "
              f"distinct values (e.g. {[round(v,3) for v in tvals[:5]]})")

    return dict(config_key=mode, output=text, nfe=nfe, tps=round(args.max_new / wall, 2),
                tokens_per_nfe=args.max_new / nfe if nfe else None,
                t_distinct=len(set(round(v, 4) for v in tvals)),
                seed_zeroed_layers=model._seed_zeroed_layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="results/abl_denoiser.jsonl")
    ap.add_argument("--modes", nargs="+",
                    default=["baseline", "freeze_time", "disable_seed"])
    ap.add_argument("--freeze-t", type=float, default=0.5)
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

    orig = type(model)._run_denoiser_step_diffusion
    type(model)._run_denoiser_step_diffusion = _ablatable_denoiser_step
    print("[ablation2/3] patched _run_denoiser_step_diffusion (freeze_time / disable_seed)")

    results = []
    for mode in args.modes:
        print(f"\n===== {mode} =====")
        for p in prompts:
            r = run(model, tok, p["prompt"], args, mode)
            print(f"  [{p['id']}] nfe={r['nfe']} tpn={r['tokens_per_nfe']:.2f}")
            results.append(dict(prompt_id=p["id"], reference=p.get("reference"), **r))

    type(model)._run_denoiser_step_diffusion = orig  # restore
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nLOGGING OK -> {args.out}  (modes={args.modes})")
    print("score with: python src/eval/gsm8k.py --in " + args.out)


if __name__ == "__main__":
    main()
