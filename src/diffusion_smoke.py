"""Coherence smoke for diffusion — passing imports is NOT enough, the output must be real
text, not word-salad. Exit 0 if the run shows genuine parallelism (NFE < max) AND low
repetition; exit 1 (garbage) otherwise. Used as the pass/fail gate at the end of install.sh.

The garbage failure mode looked like: NFE == max (nothing confident, 1 token/step) and the
output dominated by a few repeated tokens (", best best the the ..."). Both are caught here.
"""
import collections
import sys

import torch

from twotower import load, MASK_TOKEN_ID, reset_nfe, get_nfe

PROMPT = "France is a country "
MAX_NEW, BLOCK, STEPS = 64, 16, 16


def main():
    model, tok = load()  # both towers, 2 GPUs
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")

    reset_nfe(model)
    with torch.no_grad():
        out = model.generate_mask_diffusion(
            ids, max_new_tokens=MAX_NEW, block_size=BLOCK, steps_per_block=STEPS,
            mask_token_id=MASK_TOKEN_ID, temperature=0.1, confidence_threshold=0.8,
            eos_token_id=tok.eos_token_id)
    nfe = get_nfe(model) or 0
    gen_ids = out[0][ids.shape[1]:].tolist()
    text = tok.decode(gen_ids, skip_special_tokens=True)

    max_nfe = (MAX_NEW // BLOCK) * STEPS
    tokens_per_nfe = MAX_NEW / nfe if nfe else 0.0
    top_freq = (max(collections.Counter(gen_ids).values()) / len(gen_ids)) if gen_ids else 1.0

    parallel_ok = nfe < max_nfe        # a working model commits >1 token on some step
    rep_ok = top_freq < 0.35           # garbage is dominated by a few repeated tokens
    ok = parallel_ok and rep_ok

    print(f"NFE={nfe}/{max_nfe}  tokens/NFE={tokens_per_nfe:.2f}  top_token_freq={top_freq:.2f}")
    print("OUT:", repr(text[:220]))
    print("DIFFUSION", "COHERENT (PASS)" if ok else "GARBAGE (FAIL)",
          f"[parallel_ok={parallel_ok}, rep_ok={rep_ok}]")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
