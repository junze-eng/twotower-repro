"""Decisive fork: does the DENOISER-tower forward itself work, or is only the diffusion
multi-step / confidence / t logic broken?

  generate_mock_ar : uses the SAME two towers + SAME denoiser forward, but solves one token
                     at a time, deterministically. If it's coherent, the denoiser forward is
                     fine and the bug is in diffusion-specific logic (steps / confidence / t).
                     If it's ALSO garbage, the denoiser forward itself is the problem.
  generate_mask_diffusion : official HF-card params (temp=0.1, thr=0.8), 16 steps.

    python src/probe_official.py
"""
import torch

from twotower import load

model, tok = load()
ids = tok("France is a country ", return_tensors="pt").input_ids.to("cuda:0")

out = model.generate_mask_diffusion(
    ids, max_new_tokens=32, block_size=16, steps_per_block=16,
    mask_token_id=3, temperature=0.1, confidence_threshold=0.8,
    eos_token_id=tok.eos_token_id)
print("NFE:", getattr(model, "_last_nfe", None))
print("DIFFUSION:", repr(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)))

if hasattr(model, "generate_mock_ar"):
    o2 = model.generate_mock_ar(ids, max_new_tokens=32)
    print("MOCK_AR:", repr(tok.decode(o2[0][ids.shape[1]:], skip_special_tokens=True)))
else:
    print("MOCK_AR: (generate_mock_ar not available)")
