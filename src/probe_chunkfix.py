"""FIX CANDIDATE (confirmed bug): the denoiser's _denoiser_block_mamba calls
mamba_chunk_scan_combined with chunk_size=128 on a block of seq_len=16, passing
initial_states (the seeded context SSM state). Known Mamba2 SSD kernel bug (vLLM PR #21783):
when block_size < chunk_size, the initial/prev states get TOO MUCH influence (decay scale is
miscomputed) -> collapse/garbage. AR never hits this (prefill uses initial_states=None).

FIX: pass chunk_size = min(chunk_size, seq_len). chunk_size only affects performance, not
correctness (per the vLLM fix author), so this is mathematically safe and sidesteps the
block<chunk buggy path.

This monkeypatches _denoiser_block_mamba (used by BOTH the denoiser step and context extend)
and compares baseline vs fixed on a real prompt.

    python src/probe_chunkfix.py
"""
import torch
from twotower import load, MASK_TOKEN_ID

model, tok = load()
T = type(model)
orig = T._denoiser_block_mamba


def patched_block_mamba(self, mixer, hidden, init_conv, init_ssm, return_states=False):
    from einops import rearrange
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from causal_conv1d import causal_conv1d_fn

    d_inner = mixer.intermediate_size
    ngroups = mixer.n_groups
    d_state = mixer.ssm_state_size
    headdim = mixer.head_dim
    conv_dim = mixer.conv_dim
    d_conv = mixer.conv_kernel_size

    proj = mixer.in_proj(hidden)
    z, xBC, dt = torch.split(proj, [d_inner, conv_dim, mixer.num_heads], dim=-1)

    if init_conv is not None:
        init_conv = init_conv.transpose(-1, -2).contiguous().transpose(-1, -2)
    xBC_conv = causal_conv1d_fn(
        xBC.transpose(1, 2), mixer.conv1d.weight.squeeze(1), mixer.conv1d.bias,
        activation=mixer.activation, initial_states=init_conv,
    ).transpose(1, 2)

    x, B_proj, C_proj = torch.split(
        xBC_conv, [d_inner, ngroups * d_state, ngroups * d_state], dim=-1)
    x = rearrange(x, "b s (h p) -> b s h p", p=headdim).contiguous()
    B_proj = rearrange(B_proj, "b s (g n) -> b s g n", n=d_state).contiguous()
    C_proj = rearrange(C_proj, "b s (g n) -> b s g n", n=d_state).contiguous()

    _y_dtype = z.dtype
    A = -torch.exp(mixer.A_log.float())
    seq_len = x.shape[1]
    cs = min(mixer.chunk_size, seq_len)          # <-- THE FIX (avoid block<chunk bug)
    scan = mamba_chunk_scan_combined(
        x.float(), dt.float().contiguous(), A, B_proj.float(), C_proj.float(),
        cs, D=mixer.D.float(), z=None,
        dt_bias=mixer.dt_bias.float(), dt_softplus=True,
        initial_states=(init_ssm.float() if init_ssm is not None else None),
        return_final_states=return_states,
    )
    if return_states:
        y, new_ssm = scan
    else:
        y = scan
    y = rearrange(y, "b s h p -> b s (h p)").to(_y_dtype)
    y = mixer.norm(y, z)
    out = mixer.out_proj(y)
    if not return_states:
        return out
    Lx = xBC.shape[1]
    if Lx >= d_conv:
        new_conv = xBC[:, -d_conv:, :].transpose(1, 2).contiguous()
    else:
        hist = init_conv if init_conv is not None else xBC.new_zeros(xBC.shape[0], conv_dim, d_conv - 1)
        comb = torch.cat([hist.transpose(1, 2), xBC], dim=1)
        new_conv = comb[:, -d_conv:, :].transpose(1, 2).contiguous()
    return out, new_conv, new_ssm


PROMPT = ("Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many "
          "pieces do they have left in total?\nAnswer:")
ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")


def gen(tag):
    out = model.generate_mask_diffusion(
        ids, max_new_tokens=32, block_size=16, steps_per_block=16, mask_token_id=MASK_TOKEN_ID,
        temperature=0.0, confidence_threshold=0.8, eos_token_id=tok.eos_token_id)
    print(f"[{tag}] NFE={getattr(model,'_last_nfe',None)} "
          f"{tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)[:160]!r}")


T._denoiser_block_mamba = orig
gen("baseline (chunk_size=128)")
T._denoiser_block_mamba = patched_block_mamba
gen("FIXED (chunk_size=block)")
T._denoiser_block_mamba = orig
