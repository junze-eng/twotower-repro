"""Shared loading + instrumentation helpers for the TwoTower experiments.

Includes THE FIX for the diffusion-garbage bug: the HF denoiser's _denoiser_block_mamba
runs mamba_chunk_scan_combined with chunk_size=128 on a 16-token block while passing
initial_states (the seeded context SSM state). That hits the Mamba2 SSD kernel bug (vLLM
PR #21783): when block_size < chunk_size, the initial state's decay scale is miscomputed and
the seeded state over-influences the output -> denoiser collapse / word-salad. AR is fine
because it never passes initial_states. We replace the chunk-scan with a per-token
selective_state_update loop (the exact kernel AR uses successfully). Conf: fixB in
probe_fixes.py produced correct output ("...= 39..."). Set TWOTOWER_NOFIX=1 to keep the buggy
original (for a before/after demo in the writeup).
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16"
MASK_TOKEN_ID = 3  # confirmed by the HF model card


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def _fixed_denoiser_block_mamba(self, mixer, hidden, init_conv, init_ssm, return_states=False):
    """Per-token SSM scan (selective_state_update) instead of the buggy chunk-scan.
    Mathematically equivalent (denoiser Mamba is causal/forward-only) and uses AR's proven
    kernel. Conv is unchanged. Handles return_states=True (context-extend between blocks)."""
    from causal_conv1d import causal_conv1d_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update

    d_inner, ng, ds = mixer.intermediate_size, mixer.n_groups, mixer.ssm_state_size
    hd, nh = mixer.head_dim, mixer.num_heads
    conv_dim, d_conv = mixer.conv_dim, mixer.conv_kernel_size
    B_ = hidden.shape[0]

    z, xBC, dt = torch.split(mixer.in_proj(hidden), [d_inner, conv_dim, nh], -1)
    ic = init_conv.transpose(-1, -2).contiguous().transpose(-1, -2) if init_conv is not None else None
    xBC_conv = causal_conv1d_fn(
        xBC.transpose(1, 2), mixer.conv1d.weight.squeeze(1), mixer.conv1d.bias,
        activation=mixer.activation, initial_states=ic).transpose(1, 2)
    x, Bp, Cp = torch.split(xBC_conv, [d_inner, ng * ds, ng * ds], -1)
    L = x.shape[1]

    ssm = (init_ssm.float().clone() if init_ssm is not None
           else torch.zeros(B_, nh, hd, ds, device=hidden.device, dtype=torch.float32))
    A = -torch.exp(mixer.A_log.float())[:, None, None].expand(nh, hd, ds).float()
    dt_bias = mixer.dt_bias[:, None].expand(nh, hd).float()
    D = mixer.D[:, None].expand(nh, hd).float()
    ys = []
    for i in range(L):
        yi = selective_state_update(
            ssm, x[:, i].view(B_, nh, hd), dt[:, i][:, :, None].expand(B_, nh, hd),
            A, Bp[:, i].view(B_, ng, ds), Cp[:, i].view(B_, ng, ds), D,
            z=None, dt_bias=dt_bias, dt_softplus=True)
        ys.append(yi.reshape(B_, nh * hd))
    y = torch.stack(ys, 1).to(z.dtype)
    out = mixer.out_proj(mixer.norm(y, z))
    if not return_states:
        return out
    if L >= d_conv:
        new_conv = xBC[:, -d_conv:, :].transpose(1, 2).contiguous()
    else:
        hist = init_conv if init_conv is not None else xBC.new_zeros(B_, conv_dim, d_conv - 1)
        new_conv = torch.cat([hist.transpose(1, 2), xBC], dim=1)[:, -d_conv:, :].transpose(1, 2).contiguous()
    return out, new_conv, ssm


def apply_diffusion_fix(model):
    """Monkeypatch the fixed denoiser Mamba scan onto the model class (idempotent)."""
    if os.environ.get("TWOTOWER_NOFIX", "0") == "1":
        print("[twotower] TWOTOWER_NOFIX=1 -> keeping the BUGGY original chunk-scan")
        return
    type(model)._denoiser_block_mamba = _fixed_denoiser_block_mamba
    print("[twotower] applied diffusion fix (per-token selective_state_update)")


def load(ctx_device="cuda:0", den_device="cuda:1", ar_only=False):
    """Load the model, place towers, and apply the diffusion fix.

    low_cpu_mem_usage=False reads shards into RAM (fast .to(GPU)); the pod has ~2TB RAM.
    Set TWOTOWER_LOWMEM=1 to fall back to the slow mmap path on a small-RAM box.
    """
    tok = load_tokenizer()
    low_cpu = os.environ.get("TWOTOWER_LOWMEM", "0") == "1"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=low_cpu)
    if ar_only:
        model = model.to(ctx_device)
    else:
        model.place_towers_on_devices(ctx_device, den_device)
    model.eval()
    apply_diffusion_fix(model)
    return model, tok


# --- NFE (denoiser forward count) instrumentation ---------------------------
def reset_nfe(model):
    setattr(model, "_last_nfe", None)


def get_nfe(model):
    return getattr(model, "_last_nfe", None)


def nan_guard_callback(state=None):
    """A step_callback that flags NaN/Inf in the denoiser logits during generation."""
    class _CB:
        def __init__(self):
            self.hits = []
        def __call__(self, step_idx, steps_per_block, xt, t, logits, block_idx):
            if logits is not None and not torch.isfinite(logits).all():
                self.hits.append((int(block_idx), int(step_idx)))
    return _CB()
