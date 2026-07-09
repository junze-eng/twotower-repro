"""HYPOTHESIS: generate_mask_diffusion builds den_cache ONCE per block and reuses it across
all steps_per_block steps. If the denoiser Mamba path mutates those cached conv/ssm states
in place, step 2..N run on a corrupted seed -> collapse/word-salad. This also explains the
non-determinism (probe_teacher pos0 coherent vs probe_multi pos0 garbage on the same input).

TEST: force a FRESH den_cache to be built on EVERY denoiser step (no reuse). If diffusion
becomes coherent, the bug is den_cache reuse/mutation.

Two ways to force a fresh cache per step:
  A) monkeypatch _run_denoiser_step_diffusion to ignore the passed den_cache and rebuild.
This is slower (rebuild each NFE) but isolates the cause.

    python src/probe_freshcache.py
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

def gen(tag):
    out = model.generate_mask_diffusion(
        ids, max_new_tokens=32, block_size=16, steps_per_block=16, mask_token_id=3,
        temperature=0.0, confidence_threshold=0.8, eos_token_id=tok.eos_token_id)
    print(f"[{tag}] NFE={getattr(model,'_last_nfe',None)} "
          f"{tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)[:130]!r}")

# baseline (reuses den_cache per block)
type(model)._run_denoiser_step_diffusion = orig
gen("baseline-reuse")

# fresh cache every step: ignore passed den_cache -> force rebuild inside the step
def fresh(self, block_ids, cache_state, t=None, den_cache=None):
    return orig(self, block_ids, cache_state, t=t, den_cache=None)  # None -> rebuilds inside
type(model)._run_denoiser_step_diffusion = fresh
gen("fresh-each-step")
type(model)._run_denoiser_step_diffusion = orig

# also test: clone the cache's mamba states each step (cheaper isolation)
def cloned(self, block_ids, cache_state, t=None, den_cache=None):
    if den_cache is not None:
        for i in range(len(den_cache.conv_states)):
            if den_cache.conv_states[i].numel():
                den_cache.conv_states[i] = den_cache.conv_states[i].clone()
            if den_cache.ssm_states[i].numel():
                den_cache.ssm_states[i] = den_cache.ssm_states[i].clone()
    return orig(self, block_ids, cache_state, t=t, den_cache=den_cache)
type(model)._run_denoiser_step_diffusion = cloned
gen("clone-mamba-each-step")
type(model)._run_denoiser_step_diffusion = orig
