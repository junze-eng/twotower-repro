#!/usr/bin/env python3
"""
Two-tower NemotronH inference example.

Requires 2 GPUs (118GB total) for full two-tower inference.
Single GPU works for AR-only mode (context tower only, ~59GB).

Usage:
  # Mock-AR (two-tower, 2 GPUs):
  CUDA_VISIBLE_DEVICES=0,1 python inference.py --mode mock_ar

  # AR (context tower only, 1 GPU):
  python inference.py --mode ar

  # Mask diffusion (two-tower, 2 GPUs):
  python inference.py --mode mask_diffusion --model /path/to/diffusion_hf_out
"""
import argparse
import inspect
import time
import torch
import random
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from modeling_nemotron_twotower import NemotronHTwoTowerForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("prompt_arg", nargs="?", default=None)
parser.add_argument("--prompt", default=None)
parser.add_argument("--prompt-file", dest="prompt_file", default=None,
                    help="jsonl of {\"text\": ...} per line (same format as mcore "
                         "--prompt-file); each line is run as its own Request i/N.")
parser.add_argument("--model", default=str(Path(__file__).resolve().parent))
parser.add_argument("--max-new-tokens", type=int, default=128)
parser.add_argument("--mode", choices=["ar", "mock_ar", "mask_diffusion"], default="mock_ar")
parser.add_argument("--block-size", type=int, default=16)
parser.add_argument("--steps-per-block", type=int, default=16)
parser.add_argument("--mask-token-id", type=int, default=3)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=None)
parser.add_argument("--confidence-threshold", type=float, default=0.9)
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--print-diffusion-steps", action="store_true")
args = parser.parse_args()
prompt = args.prompt if args.prompt is not None else (args.prompt_arg or "France is a country ")

if args.deterministic:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = NemotronHTwoTowerForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
)

num_gpus = torch.cuda.device_count()
if num_gpus >= 2:
    # Split towers across GPUs (both towers don't fit on one 80GB card).
    # AR mode only uses the context tower (cuda:0), but placing both is fine.
    model.place_towers_on_devices("cuda:0", "cuda:1")
elif args.mode == "ar":
    # AR uses only the context tower + context head; keep the denoiser tower
    # off the GPU so a single card suffices.
    model.context_tower = model.context_tower.cuda()
    model.context_lm_head = model.context_lm_head.cuda()
else:
    model.cuda()

model.eval()
# Build the request list. A --prompt-file (jsonl, one {"text": ...} per line,
# same format mcore consumes) runs as multiple Requests i/N; otherwise the
# single positional/--prompt is the lone request.
if args.prompt_file:
    import json
    prompts = []
    with open(args.prompt_file) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["text"])
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompt_file}")
else:
    prompts = [prompt]


def step_callback(step_idx, total_steps, tokens, t=None, logits=None, block_idx=0):
    if not args.print_diffusion_steps:
        return
    if logits is None:
        print(f"\n--- Block {block_idx} Step {step_idx}/{total_steps} | init ---")
        print("xt:", tokenizer.decode(tokens[0], skip_special_tokens=False))
        return
    log_x = model._mdlm_forward(logits, tokens.to(logits.device), args.mask_token_id)
    probs = log_x.exp()[0]
    top2_probs, top2_ids = probs.topk(2, dim=-1)
    n_masked = int((tokens == args.mask_token_id).sum().item())
    print(f"\n--- Block {block_idx} Step {step_idx}/{total_steps} | masked={n_masked}/{tokens.shape[1]} | t={t:.4f} ---")
    print("xt:   " + repr(tokenizer.decode(tokens[0], skip_special_tokens=False)))
    print("top1: " + "|".join(tokenizer.decode([tid.item()])[:9].rjust(9) for tid in top2_ids[:, 0]))
    print("prb1: " + "|".join(f"{p.item():.3f}".rjust(9) for p in top2_probs[:, 0]))
    print("top2: " + "|".join(tokenizer.decode([tid.item()])[:9].rjust(9) for tid in top2_ids[:, 1]))
    print("prb2: " + "|".join(f"{p.item():.3f}".rjust(9) for p in top2_probs[:, 1]))


ctx_device = next(model.context_tower.parameters()).device
n_requests = len(prompts)
for ridx, prompt in enumerate(prompts):
    inputs = tokenizer(prompt, return_tensors="pt").to(ctx_device)
    if args.print_diffusion_steps and args.mode == "mask_diffusion":
        print(f"\n--- Diffusion steps for request {ridx + 1} ---")

    t0 = time.perf_counter()
    if args.mode == "ar":
        # Context-tower-only AR via our cached single-step path (the fair ST-AR
        # baseline). Avoids HF generate()'s cache path that crashes on this env.
        outputs = model.generate_ar(
            inputs["input_ids"], max_new_tokens=args.max_new_tokens,
            temperature=0.0, eos_token_id=tokenizer.eos_token_id,
        )
    elif args.mode == "mock_ar":
        outputs = model.generate_mock_ar(
            inputs["input_ids"], max_new_tokens=args.max_new_tokens,
            temperature=0.0, eos_token_id=tokenizer.eos_token_id,
        )
    else:
        generate_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            block_size=args.block_size,
            steps_per_block=args.steps_per_block,
            mask_token_id=args.mask_token_id,
            temperature=args.temperature,
            top_k=args.top_k,
            confidence_threshold=args.confidence_threshold,
            eos_token_id=tokenizer.eos_token_id,
        )
        if (
            args.print_diffusion_steps
            and "step_callback" in inspect.signature(model.generate_mask_diffusion).parameters
        ):
            generate_kwargs["step_callback"] = step_callback
        outputs = model.generate_mask_diffusion(inputs["input_ids"], **generate_kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - t0, 1e-9)

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[0][prompt_len:]
    n_new = int(gen_ids.shape[0])
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    nfe = getattr(model, "_last_nfe", None)

    print(f"\n--- Request {ridx + 1}/{n_requests} ---")
    print(f"Prompt: {prompt}")
    _nfe_str = f"{nfe} NFE, " if (args.mode == "mask_diffusion" and nfe is not None) else ""
    print(f"Generated ({_nfe_str}{n_new} tokens, {elapsed:.2f}s, {n_new / elapsed:.1f} tok/s):")
    print(text)

print("\n" + "=" * 70)
if args.mode == "mask_diffusion":
    print("Two-Tower mask-diffusion generation complete")
    print("=" * 70)
    print(f"  mode:                 {args.mode}")
    print(f"  block_size:           {args.block_size}")
    print(f"  steps_per_block:      {args.steps_per_block}")
    print(f"  max_new_tokens:       {args.max_new_tokens}")
    print(f"  num_blocks:           {args.max_new_tokens // args.block_size}")
    print(f"  temperature:          {args.temperature}")
    print(f"  top_k:                {args.top_k}")
    print(f"  confidence_threshold: {args.confidence_threshold}")
    print(f"  mask_token_id:        {args.mask_token_id}")
    print(f"  num_requests:         {n_requests}")
    print(f"  model:                {args.model}")
    print("=" * 70)
else:
    print("Two-tower generation complete")
    print("=" * 70)
