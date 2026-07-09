"""Is the denoiser forward actually broken, or is this just a weak base model / wrong call?
Feed a prompt whose continuation is obvious ("The capital of France is" -> " Paris") and ask
the denoiser (block all-mask, t=1.0) to predict it. If it ranks the true token near the top,
the forward WORKS (garbage is sampling/base-quality). If the true token is ranked tens of
thousands deep, the denoiser forward genuinely doesn't use the context.

    python src/probe_teacher.py
"""
import torch
from twotower import load

model, tok = load()
prompt = "The capital of France is"
cont = " Paris"
pids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")
cids = tok(cont, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda:0")

cs = model._build_context_cache(pids)
den_dev = next(model.denoiser_tower.parameters()).device
dc = model._build_denoiser_cache_diffusion(cs, den_dev)

L = 16
blk = torch.full((1, L), 3, dtype=torch.long, device=pids.device)
t = torch.tensor([1.0], device=pids.device)
lg = model._run_denoiser_step_diffusion(blk, cs, t=t, den_cache=dc)[0]  # (L, V)

tgt = int(cids[0, 0])
probs = lg[0].float().softmax(-1)
rank = int((probs > probs[tgt]).sum().item())
print(f"target {tok.decode([tgt])!r}  prob={probs[tgt]:.6f}  rank={rank}/{lg.shape[-1]}")
print("top5 pos0:", [(tok.decode([int(i)]), round(float(p), 4)) for p, i in zip(*probs.topk(5))])

# Also: compare denoiser logits vs the CONTEXT tower's own logits at that position.
ctx_logits = cs["logits"][0, -1].float()   # context tower prediction after the prompt
cprobs = ctx_logits.softmax(-1)
crank = int((cprobs > cprobs[tgt]).sum().item())
print(f"[context tower] target prob={cprobs[tgt]:.6f} rank={crank}  "
      f"top1={tok.decode([int(ctx_logits.argmax())])!r}")
