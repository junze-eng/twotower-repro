"""Test BOTH fixes for the Mamba2 SSD block<chunk initial_states bug (vLLM #21783) in ONE run.

baseline : original (mamba_chunk_scan_combined, chunk_size=128 on a 16-token block) -> garbage
fixA     : same kernel, chunk_size=min(chunk,seq_len)=16 (chunk_size doesn't affect
           correctness, only perf) -> sidesteps the block<chunk buggy branch
fixB     : replace the chunk-scan with a per-token loop of selective_state_update — the SAME
           kernel AR uses successfully. Conv unchanged. Fully avoids mamba_chunk_scan_combined.

_denoiser_block_mamba is used by BOTH the denoiser step (return_states=False) AND context
extend between blocks (return_states=True), so both patches handle return_states.

    python src/probe_fixes.py
"""
import torch
from twotower import load, MASK_TOKEN_ID

model, tok = load()
T = type(model)
ORIG = T._denoiser_block_mamba


def _conv(mixer, xBC, init_conv):
    from causal_conv1d import causal_conv1d_fn
    ic = init_conv.transpose(-1, -2).contiguous().transpose(-1, -2) if init_conv is not None else None
    return causal_conv1d_fn(
        xBC.transpose(1, 2), mixer.conv1d.weight.squeeze(1), mixer.conv1d.bias,
        activation=mixer.activation, initial_states=ic,
    ).transpose(1, 2)


def _new_conv(xBC, init_conv, conv_dim, d_conv):
    L = xBC.shape[1]
    if L >= d_conv:
        return xBC[:, -d_conv:, :].transpose(1, 2).contiguous()
    hist = init_conv if init_conv is not None else xBC.new_zeros(xBC.shape[0], conv_dim, d_conv - 1)
    comb = torch.cat([hist.transpose(1, 2), xBC], dim=1)
    return comb[:, -d_conv:, :].transpose(1, 2).contiguous()


def make_fixA():
    from einops import rearrange
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    def f(self, mixer, hidden, init_conv, init_ssm, return_states=False):
        d_inner, ng, ds = mixer.intermediate_size, mixer.n_groups, mixer.ssm_state_size
        hd, conv_dim, d_conv = mixer.head_dim, mixer.conv_dim, mixer.conv_kernel_size
        z, xBC, dt = torch.split(mixer.in_proj(hidden), [d_inner, conv_dim, mixer.num_heads], -1)
        xBC_conv = _conv(mixer, xBC, init_conv)
        x, Bp, Cp = torch.split(xBC_conv, [d_inner, ng * ds, ng * ds], -1)
        x = rearrange(x, "b s (h p) -> b s h p", p=hd).contiguous()
        Bp = rearrange(Bp, "b s (g n) -> b s g n", n=ds).contiguous()
        Cp = rearrange(Cp, "b s (g n) -> b s g n", n=ds).contiguous()
        A = -torch.exp(mixer.A_log.float())
        cs = min(mixer.chunk_size, x.shape[1])          # THE FIX
        scan = mamba_chunk_scan_combined(
            x.float(), dt.float().contiguous(), A, Bp.float(), Cp.float(), cs,
            D=mixer.D.float(), z=None, dt_bias=mixer.dt_bias.float(), dt_softplus=True,
            initial_states=(init_ssm.float() if init_ssm is not None else None),
            return_final_states=return_states)
        y, new_ssm = (scan if return_states else (scan, None))
        y = rearrange(y, "b s h p -> b s (h p)").to(z.dtype)
        out = mixer.out_proj(mixer.norm(y, z))
        if not return_states:
            return out
        return out, _new_conv(xBC, init_conv, conv_dim, d_conv), new_ssm
    return f


def make_fixB():
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update

    def f(self, mixer, hidden, init_conv, init_ssm, return_states=False):
        d_inner, ng, ds = mixer.intermediate_size, mixer.n_groups, mixer.ssm_state_size
        hd, nh, conv_dim, d_conv = mixer.head_dim, mixer.num_heads, mixer.conv_dim, mixer.conv_kernel_size
        B_ = hidden.shape[0]
        z, xBC, dt = torch.split(mixer.in_proj(hidden), [d_inner, conv_dim, nh], -1)
        xBC_conv = _conv(mixer, xBC, init_conv)          # conv unchanged (not the buggy part)
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
        return out, _new_conv(xBC, init_conv, conv_dim, d_conv), ssm
    return f


PROMPT = ("Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many "
          "pieces do they have left in total?\nAnswer:")
ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")


def gen(tag, fn):
    T._denoiser_block_mamba = fn
    try:
        out = model.generate_mask_diffusion(
            ids, max_new_tokens=32, block_size=16, steps_per_block=16, mask_token_id=MASK_TOKEN_ID,
            temperature=0.0, confidence_threshold=0.8, eos_token_id=tok.eos_token_id)
        print(f"[{tag}] NFE={getattr(model,'_last_nfe',None)} "
              f"{tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)[:160]!r}")
    except Exception as e:
        import traceback
        print(f"[{tag}] ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        T._denoiser_block_mamba = ORIG


gen("baseline", ORIG)
gen("fixA chunk=16", make_fixA())
gen("fixB per-token", make_fixB())
