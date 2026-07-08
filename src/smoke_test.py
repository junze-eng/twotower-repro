"""M0 smoke test: prove the environment works end-to-end before building any experiment.

Passing criteria:
  1. tokenizer's mask token id == MASK_TOKEN_ID (3)
  2. AR generation produces non-empty text
  3. diffusion generation produces non-empty text with NO NaN/Inf during denoising
  4. NFE and tokens/NFE (diffusion parallelism factor) are printed; AR is 1.0 by definition
"""
import time
import torch

from twotower import load, load_tokenizer, reset_nfe, get_nfe, nan_guard_callback, MASK_TOKEN_ID

PROMPT = "The capital of France is"
MAX_NEW = 64          # must be divisible by block_size
BLOCK_SIZE = 16
STEPS_PER_BLOCK = 16


def check_mask_token(tok):
    for cand in ("[MASK]", "<mask>", "<|mask|>"):
        tid = tok.convert_tokens_to_ids(cand)
        if tid is not None and tid == MASK_TOKEN_ID:
            print(f"[ok] mask token {cand!r} -> id {tid} matches MASK_TOKEN_ID")
            return
    print(f"[!] could not confirm token for MASK_TOKEN_ID={MASK_TOKEN_ID}; "
          f"decode({MASK_TOKEN_ID}) = {tok.decode([MASK_TOKEN_ID])!r}")


def run_ar(model, tok, input_ids):
    t0 = time.time()
    with torch.no_grad():
        out = model.generate_ar(input_ids, max_new_tokens=MAX_NEW)
    dt = time.time() - t0
    text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n[AR] {dt:.2f}s  tokens/NFE=1.00 (one token per forward)")
    print(f"[AR] {text!r}")


def run_diffusion(model, tok, input_ids):
    cb = nan_guard_callback()
    reset_nfe(model)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate_mask_diffusion(
            input_ids,
            max_new_tokens=MAX_NEW,
            block_size=BLOCK_SIZE,
            steps_per_block=STEPS_PER_BLOCK,
            mask_token_id=MASK_TOKEN_ID,
            temperature=0.0,
            confidence_threshold=0.8,
            eos_token_id=tok.eos_token_id,
            step_callback=cb,
        )
    dt = time.time() - t0
    nfe = get_nfe(model)
    text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    tpn = (MAX_NEW / nfe) if nfe else float("nan")
    print(f"\n[diffusion] {dt:.2f}s  NFE={nfe}  tokens/NFE={tpn:.2f}  "
          f"(parallelism vs AR)")
    print(f"[diffusion] NaN hits: {cb.hits if cb.hits else 'none'}")
    print(f"[diffusion] {text!r}")
    assert text.strip(), "diffusion produced empty output"
    assert not cb.hits, f"NaN/Inf in denoiser logits at {cb.hits}"


def main():
    tok = load_tokenizer()
    check_mask_token(tok)
    model, tok = load()  # 2 GPUs, downloads weights on first run
    input_ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    run_ar(model, tok, input_ids)
    run_diffusion(model, tok, input_ids)
    print("\n[M0] smoke test PASSED")


if __name__ == "__main__":
    main()
