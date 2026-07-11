"""Exp0 (GPU/online): run ONE diffusion generation and capture every denoising frame.

The step_callback fires once per denoising step; we snapshot the reply span (xt) each
time and dump to a compact npz. Rendering the GIF is a separate, LOCAL, GPU-free step
(exp0_render.py). This is the only part of Exp0 that needs the pod.

Defaults: max_new=64, block=16, steps=16  ->  4 blocks x 16 steps = 64 frames ("1/64").
Use the TRAINING block size (16) for a clean upper-left triangle; pass a larger
--block-size to visualize the collapse instead.
"""
import argparse
import json
import os

import numpy as np
import torch

from twotower import load, MASK_TOKEN_ID, reset_nfe, get_nfe

DEFAULT_PROMPT = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
    "clips in May. How many clips did Natalia sell altogether in April and May? Answer:"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new", type=int, default=64)      # divisible by block-size
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default="results/trace.npz")
    args = ap.parse_args()

    model, tok = load()
    input_ids = tok(args.prompt, return_tensors="pt").input_ids.to("cuda:0")
    plen = input_ids.shape[1]

    raw_frames, widths, blocks, steps, ts = [], [], [], [], []

    def cb(step_idx, steps_per_block, xt, t, logits, block_idx):
        # Capture the FULL xt row untouched. Do NOT slice by plen: at callback time xt is
        # NOT the prompt+reply sequence (that's the return value `out`), it's a narrower
        # reply/block canvas -- the old `xt[0, plen:plen+max_new]` slice ran off the end and
        # produced 0-width frames. We keep the raw row + its width + block_idx here, and let
        # the offline heatmap builder place each row correctly regardless of xt's layout.
        row = xt[0].detach().to("cpu").numpy().astype(np.int32)
        if not raw_frames:
            print(f"[cb] first xt row width={row.shape[0]}  "
                  f"(plen={plen}, max_new={args.max_new}, block_size={args.block_size})")
        raw_frames.append(row)
        widths.append(int(row.shape[0]))
        blocks.append(int(block_idx))
        steps.append(int(step_idx))
        try:
            ts.append(float(t))
        except Exception:
            ts.append(float("nan"))

    reset_nfe(model)
    with torch.no_grad():
        out = model.generate_mask_diffusion(
            input_ids,
            max_new_tokens=args.max_new,
            block_size=args.block_size,
            steps_per_block=args.steps,
            mask_token_id=MASK_TOKEN_ID,
            temperature=args.temperature,
            confidence_threshold=args.gamma,
            eos_token_id=tok.eos_token_id,
            step_callback=cb,
        )

    # Pad ragged rows to a rectangular (F, W) grid with MASK (= uncommitted/future position;
    # also stays tokenizer-decodable). block_idx / step_idx / frame_width let the offline
    # builder reconstruct the position x step triangle no matter what xt's width means.
    W = max(widths) if widths else 0
    frames = np.full((len(raw_frames), W), MASK_TOKEN_ID, dtype=np.int32)
    for i, r in enumerate(raw_frames):
        frames[i, :r.shape[0]] = r
    final = out[0, plen:plen + args.max_new].detach().cpu().numpy().astype(np.int32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta = dict(
        prompt=args.prompt, max_new=args.max_new, block_size=args.block_size,
        steps=args.steps, gamma=args.gamma, temperature=args.temperature,
        mask_token_id=MASK_TOKEN_ID, prompt_len=int(plen), nfe=get_nfe(model),
        block_idx=blocks, step_idx=steps, t=ts, frame_width=widths,
        sample_id=f"gsm8k/l{args.max_new}_b{args.block_size}_st{args.steps}_g{args.gamma}",
    )
    np.savez_compressed(args.out, frames=frames, final=final, meta=json.dumps(meta))
    print(f"saved {args.out}  frames={frames.shape}  NFE={get_nfe(model)}  "
          f"tokens/NFE={args.max_new / get_nfe(model):.2f}")
    print("decoded final:", tok.decode(final, skip_special_tokens=True))


if __name__ == "__main__":
    main()
