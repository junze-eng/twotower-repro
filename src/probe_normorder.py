"""STATIC AUDIT RESULT: the diffusion denoiser applies adaLN as modulate-THEN-norm for
mamba/attention layers (reference_modeling.py lines 656-660), while the WORKING AR path
(_forward_tower_with_cache, lines 263-271) does norm-THEN-modulate. RMSNorm after modulate
largely erases the scale term -> mangled conditioning. This patches the denoiser step to use
norm-THEN-modulate for ALL block types (like AR) and checks if diffusion becomes coherent.

Also fixes the HF_TOKEN unicode crash by not hitting the network for templates (local files
only) via local_files_only after first load.

    HF_HUB_OFFLINE=1 python src/probe_normorder.py     # weights already cached
"""
import torch
from twotower import load, MASK_TOKEN_ID
import sys

model, tok = load()
mod = sys.modules[type(model).__module__]
_get_mod_params = mod._get_mod_params
_modulate = mod._modulate

PROMPT = ("Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many "
          "pieces do they have left in total?\nAnswer:")
ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")

orig = type(model)._run_denoiser_step_diffusion


def patched(self, block_ids, cache_state, t=None, den_cache=None):
    """Same as original but norm-THEN-modulate for mamba/attention too."""
    tower = self.denoiser_tower
    den_device = next(tower.parameters()).device
    den_input = block_ids.to(den_device)
    t_emb = None
    if t is not None:
        t_dev = t.to(device=den_device, dtype=self.dtype)
        t_emb = self.t_block(self.t_embedder(t_dev))
    if den_cache is None:
        den_cache = self._build_denoiser_cache_diffusion(cache_state, den_device)
    hidden = tower.embeddings(den_input)
    for layer_idx, block in enumerate(tower.layers):
        residual = hidden
        if block.residual_in_fp32:
            residual = residual.to(torch.float32)
        mod_p = None
        if t_emb is not None:
            mod_p = _get_mod_params(t_emb, self.scale_shift_tables[layer_idx])
            shift, scale, gate = mod_p
        # PATCH: norm THEN modulate for ALL block types (match the working AR path)
        h = block.norm(hidden.to(dtype=block.norm.weight.dtype))
        if mod_p is not None:
            h = _modulate(h, shift, scale)
        if block.block_type == "mamba":
            d_conv = block.mixer.conv_kernel_size
            init_conv = den_cache.conv_states[layer_idx][..., -(d_conv - 1):]
            init_ssm = den_cache.ssm_states[layer_idx].contiguous()
            h = self._denoiser_block_mamba(block.mixer, h, init_conv, init_ssm)
        elif block.block_type == "attention":
            h = self._denoiser_block_attention(
                block.mixer, h, den_cache.key_cache[layer_idx], den_cache.value_cache[layer_idx])
        else:
            h = block.mixer(h)
        if mod_p is not None:
            h = gate.unsqueeze(1) * h
        hidden = residual + h
    hidden = tower.norm_f(hidden)
    return self.lm_head(hidden.to(self.lm_head.weight.dtype)).float()


def gen(tag):
    out = model.generate_mask_diffusion(
        ids, max_new_tokens=32, block_size=16, steps_per_block=16, mask_token_id=MASK_TOKEN_ID,
        temperature=0.0, confidence_threshold=0.8, eos_token_id=tok.eos_token_id)
    print(f"[{tag}] {tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)[:150]!r}")


type(model)._run_denoiser_step_diffusion = orig
gen("baseline (modulate-then-norm)")
type(model)._run_denoiser_step_diffusion = patched
gen("PATCHED (norm-then-modulate)")
type(model)._run_denoiser_step_diffusion = orig
