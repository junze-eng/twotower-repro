"""Localize the diffusion garbage. One run, three switchable bypasses of the denoiser-only
machinery (AR works, so the context tower / lm_head / tokenizer are fine — the bug is in
what diffusion adds). Whichever row becomes grammatical identifies the culprit.

  baseline : as-is (should be garbage)
  seed-off : Mamba initial_states=None  -> bypasses causal_conv1d/chunk-scan SEEDING kernels
                                           (the triton hypothesis lives here)
  adaln-off: time t=None                -> bypasses AdaLN modulation applied to every layer

If ALL THREE are garbage, the cause is NOT seeding/AdaLN -> suspect denoiser weights not
loading, _mdlm_forward, or cross-attention (need _denoiser_block_attention).

    python src/diagnose_diffusion.py
"""
import collections

import torch

from twotower import load, MASK_TOKEN_ID, reset_nfe, get_nfe

PROMPT = "France is a country "
MAX_NEW, BLK, STEPS = 64, 16, 16
MAX_NFE = (MAX_NEW // BLK) * STEPS


def gen(model, tok, tag):
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    reset_nfe(model)
    with torch.no_grad():
        out = model.generate_mask_diffusion(
            ids, max_new_tokens=MAX_NEW, block_size=BLK, steps_per_block=STEPS,
            mask_token_id=MASK_TOKEN_ID, temperature=0.1, confidence_threshold=0.8,
            eos_token_id=tok.eos_token_id)
    nfe = get_nfe(model) or 0
    g = out[0][ids.shape[1]:].tolist()
    text = tok.decode(g, skip_special_tokens=True)
    top = (max(collections.Counter(g).values()) / len(g)) if g else 1.0
    ok = (nfe < MAX_NFE) and (top < 0.35)
    print(f"[{tag:9}] NFE={nfe}/{MAX_NFE} tok/NFE={(MAX_NEW/nfe if nfe else 0):.2f} "
          f"top_freq={top:.2f} {'OK' if ok else 'GARBAGE'} | {text[:140]!r}")


def main():
    model, tok = load()
    T = type(model)

    gen(model, tok, "baseline")

    # seed-off: force Mamba initial_states=None (model natively supports None)
    o1 = T._denoiser_block_mamba
    T._denoiser_block_mamba = (
        lambda self, mx, h, ic, iss, return_states=False:
        o1(self, mx, h, None, None, return_states=return_states))
    gen(model, tok, "seed-off")
    T._denoiser_block_mamba = o1

    # adaln-off: force time t=None so no per-layer modulation is applied
    o2 = T._run_denoiser_step_diffusion
    T._run_denoiser_step_diffusion = (
        lambda self, block_ids, cache_state, t=None, den_cache=None:
        o2(self, block_ids, cache_state, t=None, den_cache=den_cache))
    gen(model, tok, "adaln-off")
    T._run_denoiser_step_diffusion = o2

    print("\n=> the row that turns grammatical (OK) = the bypassed module was the culprit.")
    print("   if all three are GARBAGE -> not seeding/AdaLN; suspect denoiser weights / "
          "_mdlm_forward / cross-attn (check load warnings for missing keys).")


if __name__ == "__main__":
    main()
