"""AR works on this exact 149-tok prompt but diffusion is garbage -> the bug is diffusion-
specific. probe_teacher (manual single call, t=1.0) gave coherent 'a/the/in', yet
generate's internal call is garbage. So generate feeds the denoiser something different.
This wraps _run_denoiser_step_diffusion to log EXACTLY what generate passes on the first
call (t value, xt uniqueness, block shape, cache id) and the resulting top1s — then compares
to a manual call with the same block+t.

    python src/probe_gen_internal.py
"""
import torch
from twotower import load

model, tok = load()
PROMPT = (
    "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many "
    "pieces do they have left in total?\nAnswer:"
)
ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")

orig = type(model)._run_denoiser_step_diffusion
calls = {"n": 0}

def wrap(self, block_ids, cache_state, t=None, den_cache=None):
    lg = orig(self, block_ids, cache_state, t=t, den_cache=den_cache)
    if calls["n"] < 2:
        tv = None if t is None else float(t.reshape(-1)[0])
        top1 = [tok.decode([int(lg[0, i].argmax())]) for i in range(min(8, lg.shape[1]))]
        nmask = int((block_ids == 3).sum())
        print(f"  call#{calls['n']}: t={tv} block_masked={nmask}/{block_ids.shape[1]} "
              f"den_cache_id={id(den_cache)} top1[:8]={top1}")
    calls["n"] += 1
    return lg

type(model)._run_denoiser_step_diffusion = wrap
print("=== INSIDE generate_mask_diffusion (first 2 denoiser calls) ===")
model.generate_mask_diffusion(ids, max_new_tokens=16, block_size=16, steps_per_block=4,
    mask_token_id=3, temperature=0.0, confidence_threshold=0.8, eos_token_id=tok.eos_token_id)
type(model)._run_denoiser_step_diffusion = orig

# manual reference: same all-mask block, same t=1.0, cache built the same way
print("\n=== MANUAL single call (all-mask, t=1.0) ===")
cs = model._build_context_cache(ids)
den_dev = next(model.denoiser_tower.parameters()).device
dc = model._build_denoiser_cache_diffusion(cs, den_dev)
blk = torch.full((1, 16), 3, dtype=torch.long, device=ids.device)
lg = model._run_denoiser_step_diffusion(blk, cs, t=torch.tensor([1.0], device=ids.device), den_cache=dc)
print("  manual top1[:8]:", [tok.decode([int(lg[0, i].argmax())]) for i in range(8)])
