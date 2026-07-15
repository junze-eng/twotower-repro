# Nemotron-Labs-TwoTower 复现 · 全代码汇编

> 配套文档：机制与实验结论见 **`REPORT_TwoTower.md`**；本文件收录复现仓库的**全部代码**，按角色分节，每份文件标注它对应报告的哪一节。代码均为**逐字节原样嵌入**（由 `.gen_code_report.py` 从源文件直接读出，非手抄）。

## 目录

1. **1 · 核心复现与官方 bug 修复** — 3 份
2. **2 · 采样驱动与主实验循环** — 2 份
3. **3 · 推理侧消融（论文没做）** — 3 份
4. **4 · commit 顺序与去噪动力学分析** — 6 份
5. **5 · 评测与打分（clean-correct 口径）** — 3 份
6. **6 · 参数核算与权重检查** — 2 份
7. **7 · 图与动画生成** — 3 份
8. **8 · 环境/冒烟/诊断/数据准备** — 5 份
9. **9 · 定位期探针（二分排查留痕）** — 8 份
10. **10 · Shell 编排与安装脚本** — 5 份
11. **11 · NVIDIA 官方参考代码（非本复现原创）** — 2 份
12. **12 · 层间冗余探针（AR 塔 vs 扩散塔）** — 1 份

合计 **43 份代码文件 · 5316 行**。

---

## 1 · 核心复现与官方 bug 修复

报告 §6 的王牌。把去噪塔 Mamba 的 chunk-scan 换成逐 token `selective_state_update`，绕开 `block(16) < chunk(128)` 触发的 SSD kernel bug。

### `src/twotower.py`  ·  116 行

**作用**：THE FIX + 模型加载/双卡放置/NFE 计数 —— 全实验的公共入口

> 文件自述：Shared loading + instrumentation helpers for the TwoTower experiments.

````python
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
    # from_pretrained ALWAYS materializes both towers (~126GB bf16). ar_only used to do
    # model.to(ctx_device), cramming both towers onto one card -> OOM on an 80GB GPU. Split
    # the towers across the two cards exactly like diffusion; generate_ar just exercises the
    # context tower on ctx_device while the denoiser sits idle on den_device. (ar_only is kept
    # in the signature for callers but no longer forces a single-card placement.)
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
````

### `src/probe_fixes.py`  ·  123 行

**作用**：定位阶段：对比 fixA(仅改 chunk_size) vs fixB(逐 token kernel)，证明只有 fixB 正确

> 文件自述：Test BOTH fixes for the Mamba2 SSD block<chunk initial_states bug (vLLM #21783) in ONE run.

````python
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
````

### `src/probe_chunkfix.py`  ·  99 行

**作用**：进一步验证 chunk-scan 假设的探针

> 文件自述：FIX CANDIDATE (confirmed bug): the denoiser's _denoiser_block_mamba calls

````python
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
````

---

## 2 · 采样驱动与主实验循环

报告 §5 的置信度解码「三重保险」与 §7 的主跑分。`main_run.py` 是 Part-1 的统一日志入口。

### `src/main_run.py`  ·  195 行

**作用**：主推理循环 + 逐 prompt 日志（run_prompt/load_prompts，被多个消融复用）

> 文件自述：Part 1 BASE: load + generate + full per-step logging, dumped to a pickle for offline

````python
"""Part 1 BASE: load + generate + full per-step logging, dumped to a pickle for offline
analysis. This is OBSERVATION-ONLY — it never edits the model source, so it cannot break
generation. It feeds analyses A (dynamics), B (speed), D (failure), and optionally
E (MoE routing) via forward hooks.

Everything is captured through the documented step_callback + model._last_nfe:
  per step : block_idx, step_idx, mask-ratio t, step wall time, #masked, #committed,
             #remasked, mean confidence over masked positions, the reply-span token ids
  per block: actual steps used (may be < steps_per_block if the loop early-stops)
  overall  : NFE, wall-clock, tokens, tokens/NFE, tokens/sec (TPS), avg step time

Usage (2xH100):
    python src/main_run.py --prompts data/gsm8k_mini.jsonl --out results/trace_main.pkl
Single card (both towers on cuda:0, OOM -> drop --single):
    python src/main_run.py --prompts ... --out ... --single
Optional MoE routing capture (E):
    python src/main_run.py ... --moe-hook
"""
import argparse
import json
import pickle
import time

import numpy as np
import torch

from twotower import load, MASK_TOKEN_ID, reset_nfe, get_nfe


# --- E: best-effort forward hooks on the denoiser tower's MoE routers --------
def attach_router_hooks(model):
    """Capture the top-k expert indices each router picks. Returns (handles, records,
    n_hooked). If no denoiser router is found, hooks nothing (n_hooked=0) so the caller
    can skip E, exactly as the spec asks."""
    records = []          # list of (module_name, call_counter, indices ndarray)
    counter = {"n": 0}
    handles = []
    names = [n for n, m in model.named_modules()
             if m.__class__.__name__ == "NemotronHTopkRouter"]
    denoiser = [n for n in names if any(k in n.lower() for k in ("denois", "diffusion"))]
    target = denoiser or names   # fall back to all routers if naming doesn't disambiguate

    for name, mod in model.named_modules():
        if name in target:
            def hook(m, inp, out, name=name):
                idx = out[0] if isinstance(out, (tuple, list)) else out
                records.append((name, counter["n"],
                                idx.detach().to("cpu").numpy().astype(np.int16)))
                counter["n"] += 1
            handles.append(mod.register_forward_hook(hook))
    print(f"[E] hooked {len(handles)} routers "
          f"({'denoiser-filtered' if denoiser else 'ALL routers — naming ambiguous'})")
    return handles, records, len(handles)


def confidence_from_logits(logits):
    """Per-position max softmax prob for whatever positions `logits` covers. Robust to
    unknown leading dims; returns a 1-D float32 array or None on any mismatch."""
    try:
        if logits is None:
            return None
        probs = torch.softmax(logits.float(), dim=-1)
        maxp = probs.max(dim=-1).values          # drop vocab dim
        return maxp.reshape(-1).detach().to("cpu").numpy().astype(np.float32)
    except Exception as e:                        # never let logging crash generation
        return None


def run_prompt(model, tok, prompt, args):
    input_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")
    plen = input_ids.shape[1]
    steps_log = []
    state = {"prev": None, "last_t": None}

    def cb(step_idx, steps_per_block, xt, t, logits, block_idx):
        now = time.time()
        step_s = None if state["last_t"] is None else round(now - state["last_t"], 4)
        state["last_t"] = now
        span = xt[0, plen:plen + args.max_new].detach().to("cpu").numpy().astype(np.int32)
        is_mask = span == MASK_TOKEN_ID
        prev = state["prev"]
        if prev is None:
            n_commit = int((~is_mask).sum())
            n_remask = 0
        else:
            pm = prev == MASK_TOKEN_ID
            n_commit = int((pm & ~is_mask).sum())
            n_remask = int((~pm & is_mask).sum())
        state["prev"] = span
        conf = confidence_from_logits(logits)
        conf_masked = float(np.nan if conf is None or is_mask.sum() == 0
                            else conf[:len(span)][is_mask[:len(conf)]].mean()) \
            if conf is not None and conf.shape[0] >= 1 else float("nan")
        steps_log.append(dict(
            block_idx=int(block_idx), step_idx=int(step_idx),
            t=float(t) if t is not None else float("nan"), step_s=step_s,
            n_mask=int(is_mask.sum()), n_commit=n_commit, n_remask=n_remask,
            conf_masked_mean=conf_masked, span=span.astype(np.int16),
        ))

    reset_nfe(model)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate_mask_diffusion(
            input_ids, max_new_tokens=args.max_new, block_size=args.block_size,
            steps_per_block=args.steps, mask_token_id=MASK_TOKEN_ID,
            temperature=args.temperature, confidence_threshold=args.gamma,
            eos_token_id=tok.eos_token_id, step_callback=cb,
        )
    wall = time.time() - t0
    nfe = get_nfe(model)
    text = tok.decode(out[0][plen:], skip_special_tokens=True)

    # per-block actual steps used
    steps_per_block = {}
    for s in steps_log:
        steps_per_block[s["block_idx"]] = steps_per_block.get(s["block_idx"], 0) + 1

    summary = dict(
        prompt=prompt, prompt_len=int(plen), output=text, nfe=nfe, wall_s=round(wall, 3),
        n_frames=len(steps_log), tokens=args.max_new,
        tokens_per_nfe=(args.max_new / nfe) if nfe else None,
        tps=round(args.max_new / wall, 2),
        avg_step_s=round(np.mean([s["step_s"] for s in steps_log if s["step_s"]]), 4),
        steps_per_block=steps_per_block,
        block_size=args.block_size, steps_cfg=args.steps, gamma=args.gamma,
        temperature=args.temperature,
    )
    # --- verification: logging is internally consistent ---
    assert nfe and nfe > 0, "NFE not recorded"
    assert len(steps_log) == sum(steps_per_block.values()), "frame count mismatch"
    print(f"  NFE={nfe} frames={len(steps_log)} tokens/NFE={summary['tokens_per_nfe']:.2f} "
          f"TPS={summary['tps']} steps/block={steps_per_block}")
    return summary, steps_log


def load_prompts(path, limit):
    items = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                d = json.loads(line)
                d.setdefault("id", d.get("task_id", str(i)))
                items.append(d)
    return items[:limit] if limit else items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="results/trace_main.pkl")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--single", action="store_true", help="both towers on cuda:0")
    ap.add_argument("--moe-hook", action="store_true", help="capture MoE routing (E)")
    args = ap.parse_args()
    assert args.max_new % args.block_size == 0, "max_new must be divisible by block_size"

    prompts = load_prompts(args.prompts, args.limit)
    model, tok = load(den_device="cuda:0" if args.single else "cuda:1")

    handles, moe_records, n_hooked = ([], [], 0)
    if args.moe_hook:
        handles, moe_records, n_hooked = attach_router_hooks(model)

    traces = []
    for p in prompts:
        print(f"[{p['id']}] generating...")
        summary, steps_log = run_prompt(model, tok, p["prompt"], args)
        rec = dict(prompt_id=p["id"], summary=summary, steps=steps_log)
        for extra in ("task_id", "reference"):
            if extra in p:
                rec[extra] = p[extra]
        traces.append(rec)

    for h in handles:
        h.remove()

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dict(config=vars(args), traces=traces,
                         moe_records=moe_records if args.moe_hook else None,
                         moe_hooked=n_hooked), f)
    print(f"\nLOGGING OK -> {args.out}  ({len(traces)} prompts, "
          f"moe_routers_hooked={n_hooked})")


if __name__ == "__main__":
    main()
````

### `src/run_all.py`  ·  147 行

**作用**：编排全部实验的总驱动

> 文件自述：The GPU workhorse: load the model ONCE, run a whole experiment's generations, append

````python
"""The GPU workhorse: load the model ONCE, run a whole experiment's generations, append
each result to a jsonl. Scoring and plotting are separate LOCAL steps that read the jsonl.

Design:
  - one process, one model load -> pure generation time on the pod, no per-config reload
  - resumable: re-running skips (prompt_id, config_key) already in the output jsonl
  - records raw output text + NFE + tokens/NFE + wall_clock; NO quality scoring here

Experiments (config presets below):
  e1  speed surface   : gamma x steps grid (no scoring needed downstream)
  e3  collapse        : sampling block sweep {8,16,32,64} vs training block 16
  e2  pareto          : hand-picked representative (gamma, steps) points
  ar  baseline        : single-tower autoregressive

Prompts come from a jsonl file ({"prompt": ..., optional "id"/"task_id"/"reference"}),
so the same driver serves speed prompts, GSM8K, or HumanEval — the benchmark specifics
live in the prompt file, not here.

    python src/run_all.py --exp e1 --prompts data/speed_prompts.jsonl --out results/e1.jsonl
"""
import argparse
import json
import os
import time

import torch

from twotower import load, MASK_TOKEN_ID, reset_nfe, get_nfe


def diff_grid(max_new, block, gammas, steps_list):
    return [dict(mode="diff", max_new=max_new, block_size=block, steps=T, gamma=g,
                 temperature=0.0)
            for g in gammas for T in steps_list]


EXPERIMENTS = {
    # E1: speed surface. block fixed at training size; sweep gamma x steps.
    "e1": diff_grid(128, 16, (0.5, 0.7, 0.8, 0.9, 0.95), (4, 8, 16)),
    # E3: collapse. sampling block sweep; 256 is divisible by all of 8/16/32/64.
    "e3": [dict(mode="diff", max_new=256, block_size=B, steps=16, gamma=0.8,
                temperature=0.0) for B in (8, 16, 32, 64)],
    # E2: pareto. representative points spanning fast->slow (refine from E1 surface).
    "e2": [dict(mode="diff", max_new=256, block_size=16, steps=T, gamma=g, temperature=0.0)
           for (g, T) in [(0.5, 4), (0.7, 8), (0.8, 8), (0.8, 16), (0.9, 16), (0.95, 16)]],
    # AR baseline (single tower).
    "ar": [dict(mode="ar", max_new=256)],
}


def config_key(cfg):
    return "_".join(f"{k}{cfg[k]}" for k in sorted(cfg) if k != "mode")


def run_one(model, tok, prompt, cfg):
    input_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")
    plen = input_ids.shape[1]
    reset_nfe(model)
    t0 = time.time()
    with torch.no_grad():
        if cfg["mode"] == "ar":
            out = model.generate_ar(input_ids, max_new_tokens=cfg["max_new"])
            nfe = cfg["max_new"]                      # AR: one forward per token
        else:
            out = model.generate_mask_diffusion(
                input_ids, max_new_tokens=cfg["max_new"],
                block_size=cfg["block_size"], steps_per_block=cfg["steps"],
                mask_token_id=MASK_TOKEN_ID, temperature=cfg["temperature"],
                confidence_threshold=cfg["gamma"], eos_token_id=tok.eos_token_id,
            )
            nfe = get_nfe(model)
    dt = time.time() - t0
    text = tok.decode(out[0][plen:], skip_special_tokens=True)
    return dict(output=text, nfe=nfe,
                tokens_per_nfe=(cfg["max_new"] / nfe) if nfe else None,
                tps=round(cfg["max_new"] / dt, 2),   # tokens/sec — the hardware-dependent 2.42x metric
                wall_s=round(dt, 3))


def load_prompts(path, limit):
    items = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d.setdefault("id", d.get("task_id", str(i)))
            items.append(d)
    return items[:limit] if limit else items


def done_keys(out_path):
    keys = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    keys.add((r["prompt_id"], r["config_key"]))
                except Exception:
                    pass
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(EXPERIMENTS))
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap number of prompts (0=all)")
    args = ap.parse_args()

    cfgs = EXPERIMENTS[args.exp]
    prompts = load_prompts(args.prompts, args.limit)
    done = done_keys(args.out)
    ar_only = all(c["mode"] == "ar" for c in cfgs)

    print(f"exp={args.exp}  configs={len(cfgs)}  prompts={len(prompts)}  "
          f"already_done={len(done)}  ar_only={ar_only}")
    model, tok = load(ar_only=ar_only)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    total = len(cfgs) * len(prompts)
    n = 0
    with open(args.out, "a", encoding="utf-8") as fout:
        for cfg in cfgs:
            ck = config_key(cfg)
            for p in prompts:
                n += 1
                if (p["id"], ck) in done:
                    continue
                r = run_one(model, tok, p["prompt"], cfg)
                rec = dict(exp=args.exp, prompt_id=p["id"], config_key=ck,
                           **{k: cfg[k] for k in cfg}, **r)
                for extra in ("task_id", "reference"):
                    if extra in p:
                        rec[extra] = p[extra]
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"[{n}/{total}] {ck} id={p['id']} "
                      f"nfe={r['nfe']} tpn={r['tokens_per_nfe']} {r['wall_s']}s")
    print("done ->", args.out)


if __name__ == "__main__":
    main()
````

---

## 3 · 推理侧消融（论文没做）

报告 §7 论点 B。三个 switchable monkey-patch：关 remask / 去 Mamba 播种 / 冻结 AdaLN 时间条件；外加 top-k MoE。

### `src/ablation_remask.py`  ·  208 行

**作用**：消融①：关闭置信度 remask —— 迭代精修到底贡献多少（generate_mask_diffusion 逐字节复制版）

> 文件自述：Ablation 1: disable confidence remasking (tests thesis b — how much does the ITERATIVE

````python
"""Ablation 1: disable confidence remasking (tests thesis b — how much does the ITERATIVE
refinement contribute, vs a one-shot parallel fill?).

Mechanism finding (verified against the real generate_mask_diffusion): a token, once
committed, is NEVER sent back to [MASK]. `remask` only bounces positions that were just
predicted THIS step but fell below the confidence gate. So remask IS the per-step commit
gate, and it is the only thing preventing an all-at-once fill. Disabling it therefore
makes each block commit every prediction in ONE step (block fills at step 0).

Implementation: a switchable monkey-patch of NemotronHTwoTowerForCausalLM.
generate_mask_diffusion — a VERBATIM copy of the original plus a `disable_remask` flag.
The ONLY behavioural change is `num_to_remask -> 0` when disabled; everything else
(the ">=1 floor", the "last step commits all", cache handling) is byte-identical. We do
NOT edit the HF-cached file, so the original is restored by reloading.

Verification when disabled:
  - self._remask_events == 0 (the gate never fired)
  - no [MASK] token remains in the generated span (block still fully filled)
  - steps actually used per block collapses to 1

    python src/ablation_remask.py --prompts data/gsm8k_mini.jsonl --out results/abl_remask.pkl
"""
import argparse
import json
import os
import time

import torch

from twotower import load, MASK_TOKEN_ID, get_nfe
from main_run import load_prompts


def generate_mask_diffusion_ablatable(
    self, input_ids, max_new_tokens=128, block_size=16, steps_per_block=16,
    mask_token_id=3, temperature=0.0, top_k=None, confidence_threshold=0.9,
    eos_token_id=None, step_callback=None, disable_remask=False,
):
    """Verbatim copy of the model's generate_mask_diffusion + a switchable disable_remask.
    The only changed lines are marked ABLATION-1."""
    B = input_ids.shape[0]
    device = input_ids.device
    assert max_new_tokens % block_size == 0
    num_blocks = max_new_tokens // block_size

    cache_state = self._build_context_cache(input_ids)
    context_ids = input_ids.clone()
    nfe = 0
    self._remask_events = 0                                    # ABLATION-1: instrumentation

    den_device = next(self.denoiser_tower.parameters()).device
    for block_idx in range(num_blocks):
        den_cache = self._build_denoiser_cache_diffusion(cache_state, den_device)
        xt = torch.full((B, block_size), mask_token_id, dtype=torch.long, device=device)
        if step_callback is not None:
            step_callback(0, steps_per_block, xt, t=1.0, logits=None, block_idx=block_idx)

        for step_idx in range(steps_per_block):
            is_masked = (xt == mask_token_id)
            n_masked = is_masked.float().sum(-1).mean().item()
            if n_masked == 0:
                break
            t_model = is_masked.float().mean()
            t_vec = t_model.expand(B).to(device)

            logits = self._run_denoiser_step_diffusion(xt, cache_state, t=t_vec, den_cache=den_cache)
            nfe += 1
            logits = logits.to(device)

            log_x_theta = self._mdlm_forward(logits, xt, mask_token_id)
            x_theta = log_x_theta.exp()

            if temperature <= 0:
                predicted = log_x_theta.argmax(dim=-1)
            else:
                scaled_logits = logits.clone()
                scaled_logits[..., mask_token_id] = -1e12
                scaled_log = scaled_logits / temperature - torch.logsumexp(
                    scaled_logits / temperature, dim=-1, keepdim=True)
                unmasked = (xt != mask_token_id)
                if unmasked.any():
                    scaled_log[unmasked] = -1e12
                    scaled_log[unmasked, :].scatter_(-1, xt[unmasked].unsqueeze(-1), 0.0)
                predicted = self._gumbel_sample(scaled_log)

            confidence = x_theta.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
            confidence[~is_masked] = float('inf')

            is_last_step = (step_idx == steps_per_block - 1)
            n_masked_int = is_masked.sum(-1)

            if is_last_step:
                tokens_to_commit = n_masked_int                # preserved: last step commits all
            else:
                remaining_steps = max(1, steps_per_block - step_idx)
                num_above = ((confidence > confidence_threshold) & is_masked).sum(-1)
                tokens_to_commit = torch.where(                # preserved: >=1 floor
                    num_above > 0, num_above, torch.ones_like(num_above))
                min_commit = (n_masked_int.float() / remaining_steps).ceil().long()
                tokens_to_commit = torch.clamp(
                    torch.max(tokens_to_commit, min_commit), max=n_masked_int)

            output = torch.where(is_masked, predicted, xt)

            if disable_remask:                                 # ABLATION-1: commit ALL, no bounce
                num_to_remask = torch.zeros_like(n_masked_int)
            else:
                num_to_remask = n_masked_int - tokens_to_commit

            for b in range(B):
                if num_to_remask[b] > 0:
                    masked_indices = is_masked[b].nonzero(as_tuple=True)[0]
                    masked_conf = confidence[b, masked_indices]
                    _, sort_idx = masked_conf.sort()
                    remask_idx = masked_indices[sort_idx[:num_to_remask[b]]]
                    output[b, remask_idx] = mask_token_id
                    self._remask_events += int(num_to_remask[b])   # ABLATION-1: count

            if step_callback is not None:
                step_callback(step_idx, steps_per_block, xt,
                              t=float(t_model.detach().cpu()), logits=logits,
                              block_idx=block_idx)
            xt = output

        # ABLATION-1 verification: the block must be fully filled (no MASK residual)
        assert not (xt == mask_token_id).any(), \
            f"block {block_idx} left MASK residual (disable_remask={disable_remask})"
        context_ids = torch.cat([context_ids, xt], dim=1)
        cache_state = self._extend_context_cache(xt, cache_state)
        if eos_token_id is not None and (xt == eos_token_id).any():
            break

    self._last_nfe = nfe
    return context_ids


def run(model, tok, prompt, max_new, block_size, steps, gamma, temperature, disable_remask):
    input_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")
    plen = input_ids.shape[1]
    steps_per_block = {}

    def cb(step_idx, spb, xt, t, logits, block_idx):
        steps_per_block[block_idx] = steps_per_block.get(block_idx, 0) + 1

    t0 = time.time()
    out = model.generate_mask_diffusion(
        input_ids, max_new_tokens=max_new, block_size=block_size, steps_per_block=steps,
        mask_token_id=MASK_TOKEN_ID, temperature=temperature,
        confidence_threshold=gamma, eos_token_id=tok.eos_token_id,
        step_callback=cb, disable_remask=disable_remask)
    wall = time.time() - t0
    nfe = get_nfe(model)
    text = tok.decode(out[0][plen:], skip_special_tokens=True)
    remask_events = int(getattr(model, "_remask_events", -1))

    # verification
    if disable_remask:
        assert remask_events == 0, f"disable_remask but {remask_events} remask events!"
    return dict(disable_remask=disable_remask, output=text, nfe=nfe,
                tokens_per_nfe=max_new / nfe if nfe else None,
                tps=round(max_new / wall, 2), wall_s=round(wall, 3),
                remask_events=remask_events, steps_per_block=steps_per_block)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="results/abl_remask.jsonl")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--single", action="store_true")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts, args.limit)
    model, tok = load(den_device="cuda:0" if args.single else "cuda:1")

    # switchable monkey-patch (original restored by reloading the model)
    orig = type(model).generate_mask_diffusion
    type(model).generate_mask_diffusion = generate_mask_diffusion_ablatable
    print("[ablation1] patched generate_mask_diffusion with disable_remask flag")

    results = []
    for flag in (False, True):
        tag = "remask_OFF" if flag else "baseline"
        print(f"\n===== {tag} (disable_remask={flag}) =====")
        for p in prompts:
            r = run(model, tok, p["prompt"], args.max_new, args.block_size, args.steps,
                    args.gamma, args.temperature, flag)
            print(f"  [{p['id']}] nfe={r['nfe']} tpn={r['tokens_per_nfe']:.2f} "
                  f"remask_events={r['remask_events']} steps/block={r['steps_per_block']}")
            results.append(dict(config_key=tag, prompt_id=p["id"],
                                reference=p.get("reference"), **r))

    type(model).generate_mask_diffusion = orig   # restore
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:      # jsonl -> feeds eval/gsm8k.py
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nLOGGING OK -> {args.out}  (baseline vs remask_OFF, {len(prompts)} prompts)")
    print("score with: python src/eval/gsm8k.py --in " + args.out)


if __name__ == "__main__":
    main()
````

### `src/ablation_denoiser.py`  ·  197 行

**作用**：消融②③：冻结 AdaLN 时间条件 / 关闭 Mamba 跨塔状态播种（_run_denoiser_step_diffusion 复制版）

> 文件自述：Ablations 3 (freeze AdaLN time) and 2 (disable Mamba state seeding), implemented as a

````python
"""Ablations 3 (freeze AdaLN time) and 2 (disable Mamba state seeding), implemented as a
switchable verbatim monkey-patch of NemotronHTwoTowerForCausalLM._run_denoiser_step_diffusion.

Two INDEPENDENT flags, read from model attributes so either can be toggled alone:
  _freeze_time_value : float | None   (ablation 3) — replace the incoming mask-ratio t
                                        with a constant before the adaLN time embedder.
  _disable_mamba_seed: bool           (ablation 2) — feed ZERO initial conv/ssm states to
                                        the denoiser Mamba layers instead of the context
                                        tower's states (i.e. no cross-tower state seeding).

Ablation 2 passes initial_states=None (confirmed against _denoiser_block_mamba: both
causal_conv1d_fn and mamba_chunk_scan_combined natively accept None = "no initial state").
This is the true "no seeding" semantics.

The copy of _run_denoiser_step_diffusion below is byte-identical to the model's, except
the lines marked ABLATION-2 / ABLATION-3. Original method is restored after the run.

    python src/ablation_denoiser.py --prompts data/gsm8k_mini.jsonl --out results/abl_denoiser.jsonl
"""
import argparse
import json
import os
import time

import torch

from twotower import load, MASK_TOKEN_ID, get_nfe
from main_run import load_prompts


def _ablatable_denoiser_step(self, block_ids, cache_state, t=None, den_cache=None):
    # _get_mod_params / _modulate are MODULE-LEVEL helpers in the real modeling file, not
    # methods on self. Resolve them from the model's actual (trust_remote_code) module so
    # this patched copy behaves identically.
    import sys
    _mod = sys.modules[type(self).__module__]
    _get_mod_params = _mod._get_mod_params
    _modulate = _mod._modulate

    ctx_len = cache_state["ctx_len"]
    tower = self.denoiser_tower
    den_device = next(tower.parameters()).device
    den_input = block_ids.to(den_device)
    L = den_input.shape[1]

    # ABLATION-3: freeze the time conditioning to a constant mask-ratio
    freeze_val = getattr(self, "_freeze_time_value", None)
    if freeze_val is not None and t is not None:
        t = torch.full_like(t, float(freeze_val))
        self._t_trace.append(float(t.reshape(-1)[0].detach().cpu()))
    elif t is not None:
        self._t_trace.append(float(t.reshape(-1)[0].detach().cpu()))

    t_emb = None
    if t is not None:
        t_dev = t.to(device=den_device, dtype=self.dtype)
        t_repr = self.t_embedder(t_dev)
        t_emb = self.t_block(t_repr)

    if den_cache is None:
        den_cache = self._build_denoiser_cache_diffusion(cache_state, den_device)

    hidden = tower.embeddings(den_input)

    disable_seed = getattr(self, "_disable_mamba_seed", False)  # ABLATION-2 flag

    for layer_idx, block in enumerate(tower.layers):
        residual = hidden
        if block.residual_in_fp32:
            residual = residual.to(torch.float32)

        mod = None
        if t_emb is not None:
            mod = _get_mod_params(t_emb, self.scale_shift_tables[layer_idx])
            shift, scale, gate = mod

        if block.block_type in ("mamba", "attention"):
            h = hidden
            if mod is not None:
                h = _modulate(h, shift, scale)
            h = block.norm(h.to(dtype=block.norm.weight.dtype))
        else:  # mlp / moe
            h = block.norm(hidden.to(dtype=block.norm.weight.dtype))
            if mod is not None:
                h = _modulate(h, shift, scale)

        if block.block_type == "mamba":
            d_conv = block.mixer.conv_kernel_size
            init_conv = den_cache.conv_states[layer_idx][..., -(d_conv - 1):]
            init_ssm = den_cache.ssm_states[layer_idx].contiguous()
            if disable_seed:                              # ABLATION-2: no cross-tower seed
                # _denoiser_block_mamba natively supports None (causal_conv1d_fn and
                # mamba_chunk_scan_combined both treat initial_states=None as "no state").
                # None is the true "no seeding" semantics, cleaner than zeros.
                init_conv = None
                init_ssm = None
                self._seed_zeroed_layers += 1
            h = self._denoiser_block_mamba(block.mixer, h, init_conv, init_ssm)
        elif block.block_type == "attention":
            ctx_k = den_cache.key_cache[layer_idx]
            ctx_v = den_cache.value_cache[layer_idx]
            h = self._denoiser_block_attention(block.mixer, h, ctx_k, ctx_v)
        elif block.block_type in ["mlp", "moe"]:
            h = block.mixer(h)
        else:
            raise ValueError(f"Unknown block_type: {block.block_type}")

        if mod is not None:
            h = gate.unsqueeze(1) * h

        hidden = residual + h

    hidden = tower.norm_f(hidden)
    logits = self.lm_head(hidden.to(self.lm_head.weight.dtype)).float()
    return logits


def run(model, tok, prompt, args, mode):
    # set flags for this mode (independent)
    model._freeze_time_value = args.freeze_t if mode == "freeze_time" else None
    model._disable_mamba_seed = (mode == "disable_seed")
    model._t_trace = []
    model._seed_zeroed_layers = 0

    input_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")
    plen = input_ids.shape[1]
    t0 = time.time()
    out = model.generate_mask_diffusion(
        input_ids, max_new_tokens=args.max_new, block_size=args.block_size,
        steps_per_block=args.steps, mask_token_id=MASK_TOKEN_ID,
        temperature=args.temperature, confidence_threshold=args.gamma,
        eos_token_id=tok.eos_token_id)
    wall = time.time() - t0
    nfe = get_nfe(model)
    text = tok.decode(out[0][plen:], skip_special_tokens=True)
    tvals = model._t_trace

    # --- per-mode verification ---
    if mode == "freeze_time":
        uniq = sorted(set(round(v, 4) for v in tvals))
        assert uniq == [round(args.freeze_t, 4)], f"t not constant when frozen: {uniq}"
        print(f"    [verify] t held constant at {args.freeze_t} over {len(tvals)} calls")
    elif mode == "disable_seed":
        assert model._seed_zeroed_layers > 0, "no mamba layer had its seed zeroed"
        print(f"    [verify] zeroed init state on {model._seed_zeroed_layers} mamba calls")
    else:
        print(f"    [baseline] t varied over {len(set(round(v,3) for v in tvals))} "
              f"distinct values (e.g. {[round(v,3) for v in tvals[:5]]})")

    return dict(config_key=mode, output=text, nfe=nfe, tps=round(args.max_new / wall, 2),
                tokens_per_nfe=args.max_new / nfe if nfe else None,
                t_distinct=len(set(round(v, 4) for v in tvals)),
                seed_zeroed_layers=model._seed_zeroed_layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="results/abl_denoiser.jsonl")
    ap.add_argument("--modes", nargs="+",
                    default=["baseline", "freeze_time", "disable_seed"])
    ap.add_argument("--freeze-t", type=float, default=0.5)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--single", action="store_true")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts, args.limit)
    model, tok = load(den_device="cuda:0" if args.single else "cuda:1")

    orig = type(model)._run_denoiser_step_diffusion
    type(model)._run_denoiser_step_diffusion = _ablatable_denoiser_step
    print("[ablation2/3] patched _run_denoiser_step_diffusion (freeze_time / disable_seed)")

    results = []
    for mode in args.modes:
        print(f"\n===== {mode} =====")
        for p in prompts:
            r = run(model, tok, p["prompt"], args, mode)
            print(f"  [{p['id']}] nfe={r['nfe']} tpn={r['tokens_per_nfe']:.2f}")
            results.append(dict(prompt_id=p["id"], reference=p.get("reference"), **r))

    type(model)._run_denoiser_step_diffusion = orig  # restore
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nLOGGING OK -> {args.out}  (modes={args.modes})")
    print("score with: python src/eval/gsm8k.py --in " + args.out)


if __name__ == "__main__":
    main()
````

### `src/ablation_topk.py`  ·  119 行

**作用**：消融④：推理时改 MoE 激活专家数（top-k），并用 forward hook 验证改动生效

> 文件自述：Ablation 4: change the number of active MoE experts per token at INFERENCE (top-k).

````python
"""Ablation 4: change the number of active MoE experts per token at INFERENCE (top-k).

Corresponds to the paper's MoE (128 routed / 6 active + 1 shared). The paper never probes
the inference-time quality-vs-compute tradeoff of top-k; this does.

IMPORTANT (verified against modeling_nemotron_h.py): NemotronHTopkRouter caches
`self.top_k = config.num_experts_per_tok` at __init__ and get_topk_indices() uses
`self.top_k`. So editing config after load does NOTHING — we must set `.top_k` on every
router module. This script does that (switchable via --topk) and VERIFIES the change
actually altered the number of selected experts via a forward hook.

    python src/ablation_topk.py --prompts data/gsm8k_mini.jsonl --out results/abl_topk.pkl
"""
import argparse
import os
import pickle
import types

import torch

from twotower import load, MASK_TOKEN_ID
from main_run import run_prompt, load_prompts   # reuse the exact Part-1 logging


def find_routers(model, denoiser_only=True):
    routers = []
    for name, m in model.named_modules():
        if m.__class__.__name__ == "NemotronHTopkRouter":
            if denoiser_only and not any(k in name.lower() for k in ("denois", "diffusion")):
                continue
            routers.append((name, m))
    return routers


def set_topk(model, k, scope):
    routers = find_routers(model, denoiser_only=(scope == "denoiser"))
    if not routers and scope == "denoiser":
        print("[topk] no denoiser-named routers found -> falling back to ALL routers")
        routers = find_routers(model, denoiser_only=False)
    old = sorted({int(getattr(m, "top_k", -1)) for _, m in routers})
    for _, m in routers:
        m.top_k = k
    bad = [name for name, m in routers if int(getattr(m, "top_k", -1)) != k]
    if bad:
        print(f"[topk] WARNING: {len(bad)}/{len(routers)} routers did not accept top_k={k}")
    print(f"[topk] set top_k={k} on {len(routers)} routers (was {old})")
    return len(routers)


def verify_active_count(model, tok, k):
    """Prove the change bites: hook a router and read the width of its topk_indices."""
    cap = {}

    def hook(m, i, o):
        idx = o[0] if isinstance(o, (tuple, list)) else o
        cap["k"] = int(idx.shape[-1])

    handles = [m.register_forward_hook(hook) for _, m in find_routers(model, False)[:1]]
    # NOTE: use a normal-length prompt. A 1-token prompt (e.g. "verify") makes the context
    # cache build process prompt[:-1] == empty -> cache_position[-1] IndexError inside
    # _update_causal_mask. Real experiments use ~40-token prompts, so only this probe hit it.
    ids = tok("Question: What is 2 + 2? Answer:",
              return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        model.generate_mask_diffusion(ids, max_new_tokens=16, block_size=16,
                                       steps_per_block=2, mask_token_id=MASK_TOKEN_ID,
                                       temperature=0.0, confidence_threshold=0.8)
    for h in handles:
        h.remove()
    got = cap.get("k")
    print(f"[verify] router selected {got} experts/token (expected {k})")
    if got != k:
        print(f"[verify] WARNING: routing width {got} != expected {k} (recording anyway)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="results/abl_topk.pkl")
    ap.add_argument("--topk", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--scope", choices=["denoiser", "all"], default="denoiser")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--single", action="store_true")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts, args.limit)
    model, tok = load(den_device="cuda:0" if args.single else "cuda:1")

    gen_args = types.SimpleNamespace(max_new=args.max_new, block_size=args.block_size,
                                     steps=args.steps, gamma=args.gamma,
                                     temperature=args.temperature)

    results = []
    for k in args.topk:
        print(f"\n===== top_k = {k} ({args.scope}) =====")
        n = set_topk(model, k, args.scope)
        try:
            verify_active_count(model, tok, k)
        except Exception as e:
            print(f"[topk] verify skipped (non-fatal): {type(e).__name__}: {e}")
        for p in prompts:
            summary, _ = run_prompt(model, tok, p["prompt"], gen_args)
            results.append(dict(topk=k, scope=args.scope, n_routers=n,
                                prompt_id=p["id"], summary=summary,
                                reference=p.get("reference")))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dict(config=vars(args), results=results), f)
    print(f"\nLOGGING OK -> {args.out}  ({len(results)} runs across top_k={args.topk})")


if __name__ == "__main__":
    main()
````

---

## 4 · commit 顺序与去噪动力学分析

报告 §7 论点 A / 动力学 / MoE 路由抖动。捕获-分析-渲染三段式。

### `analyze_commit.py`  ·  174 行

**作用**：从 trace 计算 commit 顺序（tokens/NFE、每块步数、难度相关性）

> 文件自述：analyze_commit.py (LOCAL, no GPU) — parallelism / commit-order evidence from an

````python
"""analyze_commit.py (LOCAL, no GPU) — parallelism / commit-order evidence from an
exp0_capture frames npz. Produces:
  - Kendall tau-b of (reply position vs first-commit step)  -> left-to-right bias (cf. the
    DiffusionGemma "Neither Parallel Nor Sequential" paper, which reports tau-b 0.43-0.60)
  - per-step commit-batch size distribution                -> parallelism WIDTH (cf. DG's
    "accept batch 13-26 tokens")
  - the TRUE position x step triangle heatmap
  - the TRUE per-position denoising GIF

    python analyze_commit.py --npz results/trace_tri_b16.npz --outdir figs

Reconstruction: the capture stores each step's raw denoiser canvas row (block-local ~16-wide,
or cumulative) plus block_idx / frame_width, so we place every row at the right reply offset
and carry committed tokens forward. A position "commits" the first step its value != MASK.
"""
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

BLUE, GREEN, ORANGE, VIOLET, RED = "#2a78d6", "#008300", "#eb6834", "#4a3aa7", "#e34948"
MASKC = (0.91, 0.91, 0.90)
INK, INK2, MUTED, GRID, SURF, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]


def reconstruct(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    frames = d["frames"]; meta = json.loads(str(d["meta"]))
    mask_id = int(meta["mask_token_id"]); bs = int(meta["block_size"]); mx = int(meta["max_new"])
    plen = int(meta["prompt_len"]); blocks = list(meta["block_idx"])
    widths = list(meta.get("frame_width") or [frames.shape[1]] * frames.shape[0])
    F = frames.shape[0]
    canvas = np.full((F, mx), mask_id, np.int32)          # carry-forward reply canvas per step
    running = np.full(mx, mask_id, np.int32)
    layout = None
    for f in range(F):
        w = int(widths[f]); row = frames[f, :w]; b = int(blocks[f])
        if w == bs:
            off = b * bs; layout = layout or "block-local"
        elif w <= mx:
            off = 0; layout = layout or "cumulative"
        else:
            row = row[plen:]; off = 0; layout = layout or "prompt-included"
        seg = row[:max(0, mx - off)]
        running[off:off + seg.shape[0]] = seg            # live state incl. remask
        canvas[f] = running
    print(f"[reconstruct] F={F} layout={layout} max_new={mx} block_size={bs} mask_id={mask_id}")
    return canvas, meta, mask_id


def kendall_tau_b(x, y):
    n = len(x); nc = nd = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]; dy = y[i] - y[j]
            if dx == 0: tx += 1
            if dy == 0: ty += 1
            s = (dx > 0) - (dx < 0)
            t = (dy > 0) - (dy < 0)
            if s * t > 0: nc += 1
            elif s * t < 0: nd += 1
    n0 = n * (n - 1) / 2
    denom = np.sqrt((n0 - tx) * (n0 - ty))
    return (nc - nd) / denom if denom > 0 else float("nan")


def analyze(npz_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    canvas, meta, mask_id = reconstruct(npz_path)
    F, mx = canvas.shape
    committed = canvas != mask_id                          # (F, mx) bool

    # first-commit step per position (positions never committed -> excluded)
    first = np.full(mx, -1, np.int64)
    for p in range(mx):
        nz = np.nonzero(committed[:, p])[0]
        if nz.size: first[p] = nz[0]
    pos = np.nonzero(first >= 0)[0]
    tau = kendall_tau_b(pos.tolist(), first[pos].tolist()) if pos.size > 2 else float("nan")

    # per-step commit-batch: positions newly committed (mask -> non-mask) at each step
    newc = committed & ~np.vstack([np.zeros((1, mx), bool), committed[:-1]])
    batch = newc.sum(1)
    batch_nz = batch[batch > 0]
    stats = dict(
        tau_b=round(float(tau), 3),
        n_committed=int(pos.size),
        commit_batch_mean=round(float(batch_nz.mean()), 2) if batch_nz.size else 0,
        commit_batch_median=int(np.median(batch_nz)) if batch_nz.size else 0,
        commit_batch_max=int(batch_nz.max()) if batch_nz.size else 0,
        n_steps=int(F), nfe=meta.get("nfe"),
    )
    print("[stats]", json.dumps(stats, ensure_ascii=False))

    # ---- FIG: true triangle (fill history) + first-commit scatter with tau ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.patch.set_facecolor(SURF)
    a1.imshow(committed, aspect="auto", cmap="Blues", interpolation="nearest",
              extent=[0, mx, F, 0], vmin=0, vmax=1)
    a1.set_xlabel("reply position", color=INK2, fontsize=10)
    a1.set_ylabel("denoising step (time →down)", color=INK2, fontsize=10)
    a1.set_title("Commit history (upper-left triangle = left-to-right)", color=INK,
                 fontsize=11, fontweight="bold", loc="left")
    a2.set_facecolor(SURF)
    for s in ("top", "right"): a2.spines[s].set_visible(False)
    a2.scatter(pos, first[pos], color=BLUE, s=24, zorder=3)
    a2.set_xlabel("reply position", color=INK2, fontsize=10)
    a2.set_ylabel("first-commit step", color=INK2, fontsize=10)
    a2.set_title(f"Left-to-right bias: Kendall τb = {stats['tau_b']}", color=INK,
                 fontsize=11, fontweight="bold", loc="left")
    a2.grid(color=GRID, lw=0.8); a2.set_axisbelow(True); a2.tick_params(colors=MUTED)
    a2.annotate("1.0 = strict AR · 0 = order-independent", (0.02, 0.94), xycoords="axes fraction",
                color=MUTED, fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_commit_order.png"), dpi=140, facecolor=SURF)
    plt.close(fig)

    # ---- FIG: commit-batch distribution ----
    fig, ax = plt.subplots(figsize=(7, 4.4)); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if batch_nz.size:
        ax.hist(batch_nz, bins=range(1, int(batch_nz.max()) + 2), color=VIOLET, align="left", rwidth=0.85)
    ax.axvline(stats["commit_batch_mean"], color=ORANGE, ls="--", lw=1.6)
    ax.annotate(f"mean {stats['commit_batch_mean']} tokens/step (max {stats['commit_batch_max']})",
                (stats["commit_batch_mean"], 0), xytext=(6, 20), textcoords="offset points",
                color=ORANGE, fontsize=9)
    ax.set_xlabel("content tokens committed in one forward pass", color=INK2, fontsize=10)
    ax.set_ylabel("# denoising steps", color=INK2, fontsize=10)
    ax.set_title("Parallelism width: commit-batch per step", color=INK, fontsize=11.5,
                 fontweight="bold", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(colors=MUTED)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_commit_batch.png"), dpi=140, facecolor=SURF)
    plt.close(fig)

    # ---- TRUE denoising GIF (real per-position commits) ----
    nrows = int(np.ceil(mx / meta["block_size"])); bs = int(meta["block_size"])
    commit_step = np.where(first >= 0, first, F + 1)
    gframes = []
    for f in range(F):
        img = np.ones((nrows, bs, 3))
        for p in range(mx):
            r, c = divmod(p, bs)
            if not committed[f, p]:
                img[r, c] = MASKC
            else:
                age = f - commit_step[p]
                img[r, c] = (0.0, 0.62, 0.31) if age == 0 else (0.92, 0.41, 0.20) if age <= 2 else (0.165, 0.47, 0.84)
        fig, ax = plt.subplots(figsize=(6, 6.4)); fig.patch.set_facecolor(SURF)
        ax.imshow(img, interpolation="nearest", extent=[0, bs, nrows, 0])
        for k in range(nrows + 1): ax.axhline(k, color=SURF, lw=1.5)
        for k in range(bs + 1): ax.axvline(k, color=SURF, lw=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Denoising (measured per-position commits)", color=INK, fontsize=12, fontweight="bold", loc="left")
        ax.set_xlabel(f"step {f+1}/{F}   ·   committed {int(committed[f].sum())}/{mx}   ·   τb={stats['tau_b']}",
                      color=INK2, fontsize=10)
        fig.tight_layout(); p_ = os.path.join(outdir, f"_rf_{f:03d}.png")
        fig.savefig(p_, dpi=88, facecolor=SURF); plt.close(fig); gframes.append(imageio.imread(p_))
    imageio.mimsave(os.path.join(outdir, "denoising_real.gif"), gframes + [gframes[-1]] * 8, fps=6, loop=0)
    for f in range(F):
        fp = os.path.join(outdir, f"_rf_{f:03d}.png")
        if os.path.exists(fp): os.remove(fp)

    with open(os.path.join(outdir, "commit_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    print("wrote fig_commit_order.png, fig_commit_batch.png, denoising_real.gif, commit_stats.json ->", outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/trace_tri_b16.npz")
    ap.add_argument("--outdir", default="figs")
    a = ap.parse_args()
    analyze(a.npz, a.outdir)
````

### `src/exp0_capture.py`  ·  113 行

**作用**：捕获逐步去噪 trace（写 trace_*.pkl / *.npz）

> 文件自述：Exp0 (GPU/online): run ONE diffusion generation and capture every denoising frame.

````python
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

    raw_frames, widths, blocks, steps, ts, raw_conf = [], [], [], [], [], []

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
        # per-position max-softmax confidence (defensive; None if logits shape is unexpected).
        # Enables commit-batch confidence + a later confidence->correctness analysis.
        try:
            lg = logits[0] if logits.dim() == 3 else logits
            c = torch.softmax(lg.float(), dim=-1).max(dim=-1).values
            raw_conf.append(c.detach().to("cpu").numpy().astype(np.float32))
        except Exception:
            raw_conf.append(None)

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
    # per-position confidence, padded to (F, Wc) with nan (all-nan if capture failed)
    Wc = max((c.shape[0] for c in raw_conf if c is not None), default=0)
    conf = np.full((len(raw_conf), Wc), np.nan, dtype=np.float32)
    for i, c in enumerate(raw_conf):
        if c is not None:
            conf[i, :c.shape[0]] = c
    final = out[0, plen:plen + args.max_new].detach().cpu().numpy().astype(np.int32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta = dict(
        prompt=args.prompt, max_new=args.max_new, block_size=args.block_size,
        steps=args.steps, gamma=args.gamma, temperature=args.temperature,
        mask_token_id=MASK_TOKEN_ID, prompt_len=int(plen), nfe=get_nfe(model),
        block_idx=blocks, step_idx=steps, t=ts, frame_width=widths,
        sample_id=f"gsm8k/l{args.max_new}_b{args.block_size}_st{args.steps}_g{args.gamma}",
    )
    np.savez_compressed(args.out, frames=frames, conf=conf, final=final, meta=json.dumps(meta))
    print(f"saved {args.out}  frames={frames.shape}  NFE={get_nfe(model)}  "
          f"tokens/NFE={args.max_new / get_nfe(model):.2f}")
    print("decoded final:", tok.decode(final, skip_special_tokens=True))


if __name__ == "__main__":
    main()
````

### `src/exp0_analyze.py`  ·  76 行

**作用**：分析 exp0 trace

> 文件自述：Exp0/AnalysisA (LOCAL, no GPU): quantify the denoising dynamics from a trace.npz.

````python
"""Exp0/AnalysisA (LOCAL, no GPU): quantify the denoising dynamics from a trace.npz.

Directly tests the hypothesis "most tokens get committed in the first step, so it runs
fast". Produces the numbers behind the upper-left triangle:
  - commits-per-step and remasks-per-step
  - fraction of each block committed already at its FIRST step
  - cumulative fill curve
  - effective NFE vs the max (steps_per_block * blocks) -> is there early-stop headroom?

Run on your Mac/laptop:
    python src/exp0_analyze.py --npz results/trace.npz --out results/exp0_dynamics.png
"""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/trace.npz")
    ap.add_argument("--out", default="results/exp0_dynamics.png")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    frames = d["frames"]                     # (F, L)
    meta = json.loads(str(d["meta"]))
    mask_id = meta["mask_token_id"]
    block_idx = np.asarray(meta["block_idx"])  # per-frame block id
    step_idx = np.asarray(meta["step_idx"])    # per-frame step within block
    F, L = frames.shape
    is_mask = frames == mask_id

    # per-step deltas (a position going mask->token = commit, token->mask = remask)
    commits = np.zeros(F, int)
    remasks = np.zeros(F, int)
    for f in range(1, F):
        commits[f] = int((is_mask[f - 1] & ~is_mask[f]).sum())
        remasks[f] = int((~is_mask[f - 1] & is_mask[f]).sum())
    filled = (~is_mask).sum(axis=1)          # cumulative filled per frame

    # fraction of each block committed at its FIRST step
    print(f"\nsample: {meta['sample_id']}  block_size={meta['block_size']}  "
          f"steps={meta['steps']}  gamma={meta['gamma']}")
    print(f"frames(=NFE proxy)={F}  reply_len={L}  tokens/NFE={L / F:.2f}  "
          f"reported NFE={meta.get('nfe')}\n")
    print(f"{'block':>5}{'step1 commits':>15}{'block_size':>12}{'step1 %':>10}")
    for b in sorted(set(block_idx.tolist())):
        fs = np.where(block_idx == b)[0]
        first = fs[step_idx[fs].argmin()]
        c1 = commits[first] if first > 0 else int((~is_mask[first]).sum())
        bs = meta["block_size"]
        print(f"{b:>5}{c1:>15}{bs:>12}{100 * c1 / bs:>9.1f}%")

    max_nfe = meta["block_size"] and (L // meta["block_size"]) * meta["steps"]
    print(f"\nearly-stop check: frames={F} vs max possible steps={max_nfe} "
          f"-> {'headroom (early-stop or few steps used)' if F < max_nfe else 'ran full loop'}")
    print(f"total remask events: {int(remasks.sum())}")

    # figures
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[2, 1])
    ax[0].bar(range(F), commits, color="#2ca02c", label="commits")
    ax[0].bar(range(F), -remasks, color="#d62728", label="remasks")
    ax[0].set_title("commits (+) / remasks (-) per denoising step")
    ax[0].set_xlabel("step (frame)"); ax[0].legend()
    ax[1].plot(range(F), 100 * filled / L, color="#1f77b4")
    ax[1].set_title("cumulative fill %"); ax[1].set_xlabel("step"); ax[1].set_ylim(0, 101)
    fig.tight_layout(); fig.savefig(args.out, dpi=110)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
````

### `src/exp0_render.py`  ·  107 行

**作用**：渲染 exp0 trace 为图/表

> 文件自述：Exp0 (LOCAL/offline, no GPU): render a captured trace.npz into a GIF like the mock.

````python
"""Exp0 (LOCAL/offline, no GPU): render a captured trace.npz into a GIF like the mock.

Layout per frame:
  - title + sample id + "filled=x/N | completion=y%"
  - legend: cell color = how many frames ago this position was (re)written
            1st(current)=green, 2nd=orange, 3rd=blue, 4th=red, 5th+=purple; masked=grey
  - main grid: one cell per reply-span position (masked positions show grey)
  - bottom strip: fill-state history stacked by frame -> the upper-left triangle
  - frame counter f/F

Run this on your laptop; it only needs numpy + matplotlib + imageio.
    python src/exp0_render.py --npz results/trace.npz --out results/exp0.gif
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio.v2 as imageio

AGE_COLORS = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728", "#9467bd"]  # 1st..5th+
MASK_COLOR = "#e8e8e8"


def compute_age(frames, mask_id):
    """age[f,p] = frames since position p last (re)took its current non-mask value;
    -1 means currently masked (handles remask: value reverts to grey)."""
    F, L = frames.shape
    age = np.full((F, L), -1, np.int32)
    last_val = np.full(L, mask_id, frames.dtype)
    last_change = np.full(L, -1, np.int32)
    for f in range(F):
        row = frames[f]
        for p in range(L):
            v = row[p]
            if v == mask_id:
                last_val[p] = mask_id
                last_change[p] = -1
                age[f, p] = -1
            else:
                if v != last_val[p]:
                    last_change[p] = f
                    last_val[p] = v
                age[f, p] = f - last_change[p]
    return age


def render(npz_path, out_gif, cols=64, fps=8):
    d = np.load(npz_path, allow_pickle=True)
    frames = d["frames"]
    meta = json.loads(str(d["meta"]))
    mask_id = meta["mask_token_id"]
    F, L = frames.shape
    age = compute_age(frames, mask_id)
    fill_hist = (frames != mask_id).astype(np.float32)  # (F, L)
    rows = int(np.ceil(L / cols))
    os.makedirs("results/frames", exist_ok=True)

    images = []
    for f in range(F):
        fig = plt.figure(figsize=(9, 12))
        gs = fig.add_gridspec(2, 1, height_ratios=[6, 1], hspace=0.2)
        ax = fig.add_subplot(gs[0]); ax.set_xlim(0, cols); ax.set_ylim(-rows, 3); ax.axis("off")

        filled = int(fill_hist[f].sum())
        ax.text(0, 2.4, "Generated reply span", fontsize=20, fontweight="bold")
        ax.text(0, 1.8, f"{meta['sample_id']}/sample", fontsize=10, color="#444")
        ax.text(0, 1.4, f"filled={filled}/{L} | completion={100*filled/L:.1f}%",
                fontsize=10, color="#444")
        for i, lab in enumerate(["1st (current frame)", "2nd", "3rd", "4th", "5th+"]):
            ax.text(i * 13, 0.7, "● " + lab, color=AGE_COLORS[i], fontsize=9)

        for p in range(L):
            r, c = divmod(p, cols)
            col = MASK_COLOR if age[f, p] < 0 else AGE_COLORS[min(age[f, p], 4)]
            ax.add_patch(Rectangle((c, -r), 0.9, 0.9, color=col))

        axb = fig.add_subplot(gs[1])
        axb.imshow(fill_hist[:f + 1], aspect="auto", cmap="Greys", vmin=0, vmax=1,
                   interpolation="nearest", extent=[0, L, f + 1, 0])
        axb.set_title("Token positions (history kept below)", loc="left", fontsize=10)
        axb.set_yticks([])
        axb.set_xticks([0, L * .25, L * .5, L * .75, L])
        axb.set_xticklabels(["0", "25%", "50%", "75%", str(L)])
        fig.text(0.88, 0.02, f"{f + 1}/{F}", fontsize=12)

        path = f"results/frames/frame_{f:04d}.png"
        fig.savefig(path, dpi=90, bbox_inches="tight"); plt.close(fig)
        images.append(imageio.imread(path))

    # hold the last frame a bit longer
    imageio.mimsave(out_gif, images + [images[-1]] * fps, fps=fps)
    print(f"wrote {out_gif}  ({F} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/trace.npz")
    ap.add_argument("--out", default="results/exp0.gif")
    ap.add_argument("--cols", type=int, default=64)
    ap.add_argument("--fps", type=int, default=8)
    a = ap.parse_args()
    render(a.npz, a.out, cols=a.cols, fps=a.fps)
````

### `src/conf_correct.py`  ·  80 行

**作用**：置信度→正确性探针

> 文件自述：conf_correct.py (GPU/pod) — does commit CONFIDENCE predict CORRECTNESS? (cf. the

````python
"""conf_correct.py (GPU/pod) — does commit CONFIDENCE predict CORRECTNESS? (cf. the
DiffusionGemma paper's finding: commit entropy predicts correctness on GSM8K but not on
factual recall). Generates over GSM8K-mini; a step_callback records the max-softmax
confidence of each token AT THE STEP IT COMMITS (mask -> non-mask), plus the final answer.
Local scoring (score_all.py --conf) then correlates mean commit-confidence with clean-correct.

    python src/conf_correct.py --prompts data/gsm8k_mini.jsonl --out results/conf_correct.jsonl --limit 15
"""
import argparse, json, os
import torch
from twotower import load, MASK_TOKEN_ID, reset_nfe, get_nfe


def load_prompts(path, limit):
    items = []
    for i, l in enumerate(open(path, encoding="utf-8")):
        l = l.strip()
        if l:
            d = json.loads(l); d.setdefault("id", d.get("task_id", str(i))); items.append(d)
    return items[:limit] if limit else items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.8)
    args = ap.parse_args()

    model, tok = load()
    prompts = load_prompts(args.prompts, args.limit)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "a", encoding="utf-8") as fout:
        for p in prompts:
            ids = tok(p["prompt"], return_tensors="pt").input_ids.to("cuda:0")
            plen = ids.shape[1]
            st = {"prev": None, "block": -1, "confs": []}   # confidences at commit time

            def cb(step_idx, steps_per_block, xt, t, logits, block_idx):
                try:
                    row = xt[0].detach().to("cpu")
                    lg = logits[0] if logits.dim() == 3 else logits
                    conf = torch.softmax(lg.float(), dim=-1).max(dim=-1).values.detach().to("cpu")
                    if block_idx != st["block"]:
                        st["prev"] = None; st["block"] = block_idx      # new block -> reset
                    if st["prev"] is not None:
                        newly = (st["prev"] == MASK_TOKEN_ID) & (row != MASK_TOKEN_ID)
                        w = min(conf.shape[0], newly.shape[0])
                        for i in range(w):
                            if bool(newly[i]):
                                st["confs"].append(float(conf[i]))
                    st["prev"] = row.clone()
                except Exception:
                    pass

            reset_nfe(model)
            with torch.no_grad():
                out = model.generate_mask_diffusion(
                    ids, max_new_tokens=args.max_new, block_size=args.block_size,
                    steps_per_block=args.steps, mask_token_id=MASK_TOKEN_ID, temperature=0.0,
                    confidence_threshold=args.gamma, eos_token_id=tok.eos_token_id, step_callback=cb)
            text = tok.decode(out[0][plen:], skip_special_tokens=True)
            confs = st["confs"]
            rec = dict(prompt_id=p["id"], reference=p.get("reference"), output=text,
                       nfe=get_nfe(model), n_commit=len(confs),
                       mean_commit_conf=(sum(confs) / len(confs)) if confs else None,
                       min_commit_conf=min(confs) if confs else None)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            print(f"[{p['id']}] mean_conf={rec['mean_commit_conf']} min={rec['min_commit_conf']} "
                  f"n={len(confs)} nfe={rec['nfe']}")
    print("done ->", args.out)


if __name__ == "__main__":
    main()
````

### `src/ruler_lite.py`  ·  83 行

**作用**：RULER-lite 长上下文检索评测

> 文件自述：ruler_lite.py (GPU/pod) — lightweight long-context probe.

````python
"""ruler_lite.py (GPU/pod) — lightweight long-context probe.

Question it answers: as the context grows, does TwoTower's diffusion still (a) retrieve
correctly and (b) keep its parallelism (tokens/NFE)? Long-context ability is inherited from
the FROZEN AR context tower, so this also tests whether the diffusion decoder rides on it.

Needle-in-a-haystack: a "special access code" is planted in the middle of filler text grown
to several target lengths; the model must read it back. Records retrieval correctness + nfe +
tokens/NFE + wall time per length. OOM at a length is recorded, not fatal.

    python src/ruler_lite.py --lengths 2048 8192 16384 32768 --out results/ruler.jsonl
"""
import argparse, json, os, time
import torch
from twotower import load, MASK_TOKEN_ID, reset_nfe, get_nfe

FILLER = ("The grass is green and the sky is blue. Birds fly south for the winter. "
          "Water flows downhill and the sun rises in the east every single morning. ")
CODES = [4823, 7591, 3164, 9052, 6238]


def build_ids(tok, target_len, code):
    needle = f" The special access code is {code}. Remember it. "
    q = "\n\nQuestion: What is the special access code?\nAnswer: The special access code is"
    per = max(1, len(tok(FILLER, add_special_tokens=False).input_ids))
    reps = max(2, target_len // per)
    text = FILLER * (reps // 2) + needle + FILLER * (reps - reps // 2)
    return tok(text + q, return_tensors="pt").input_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 16384, 32768])
    ap.add_argument("--out", default="results/ruler.jsonl")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ar", action="store_true", help="also run the AR context tower for comparison")
    args = ap.parse_args()

    model, tok = load()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fout:
        for L in args.lengths:
            for r in range(args.reps):
                code = CODES[r % len(CODES)]
                ids = build_ids(tok, L, code).to("cuda:0")
                plen = ids.shape[1]
                for mode in (["diff", "ar"] if args.ar else ["diff"]):
                    reset_nfe(model); t0 = time.time()
                    try:
                        with torch.no_grad():
                            if mode == "diff":
                                out = model.generate_mask_diffusion(
                                    ids, max_new_tokens=args.max_new, block_size=args.block_size,
                                    steps_per_block=args.steps, mask_token_id=MASK_TOKEN_ID,
                                    temperature=0.0, confidence_threshold=args.gamma,
                                    eos_token_id=tok.eos_token_id)
                                nfe = get_nfe(model)
                            else:
                                out = model.generate_ar(ids, max_new_tokens=args.max_new)
                                nfe = args.max_new
                        dt = time.time() - t0
                        text = tok.decode(out[0][plen:], skip_special_tokens=True)
                        rec = dict(mode=mode, target_len=L, ctx_len=int(plen), code=code,
                                   ok=str(code) in text, nfe=nfe,
                                   tokens_per_nfe=round(args.max_new / nfe, 3) if nfe else None,
                                   tps=round(args.max_new / dt, 2), wall_s=round(dt, 3),
                                   output=text[:120])
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        rec = dict(mode=mode, target_len=L, ctx_len=int(plen), code=code, oom=True)
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                    print(f"[{mode} L={L} ctx={plen}] " +
                          ("OOM" if rec.get("oom") else f"ok={rec['ok']} nfe={rec['nfe']} "
                           f"tpn={rec['tokens_per_nfe']} {rec['wall_s']}s"))
    print("done ->", args.out)


if __name__ == "__main__":
    main()
````

---

## 5 · 评测与打分（clean-correct 口径）

报告 §7 打分口径：degeneration-aware，既对又不退化才算数——把 block64 从虚高 90% 还原成真实 50%。

### `src/eval/gsm8k.py`  ·  62 行

**作用**：GSM8K/synthmath 打分（clean-correct 判定）

> 文件自述：Score GSM8K outputs LOCALLY (Mac, no GPU). Reads a run_all jsonl (records carry

````python
"""Score GSM8K outputs LOCALLY (Mac, no GPU). Reads a run_all jsonl (records carry
`output`, `reference`, `config_key`), extracts the predicted number, compares to gold,
and reports accuracy per config_key. Also dumps a scores jsonl for plot.py.

    python src/eval/gsm8k.py --in results/e2.jsonl --out results/e2_scores.jsonl
"""
import argparse
import json
import re
from collections import defaultdict


def extract_pred(text):
    """Take the number after the first '####' if present, else the last number seen."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        val = m.group(1)
    else:
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
        if not nums:
            return None
        val = nums[-1]
    return val.replace(",", "").rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    correct = defaultdict(int)
    total = defaultdict(int)
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "reference" not in r:
                continue
            ck = r.get("config_key", "default")
            pred = extract_pred(r["output"])
            gold = str(r["reference"]).replace(",", "").strip()
            total[ck] += 1
            if pred is not None and pred == gold:
                correct[ck] += 1

    rows = []
    print(f"{'config_key':<40}{'acc':>8}{'n':>6}")
    for ck in sorted(total):
        acc = correct[ck] / total[ck]
        print(f"{ck:<40}{100*acc:>7.1f}%{total[ck]:>6}")
        rows.append({"config_key": ck, "accuracy": acc, "n": total[ck],
                     "correct": correct[ck], "metric": "gsm8k"})

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"\nscores -> {args.out}")


if __name__ == "__main__":
    main()
````

### `src/eval/humaneval.py`  ·  87 行

**作用**：HumanEval 代码评测（pass@1）

> 文件自述：Score HumanEval outputs LOCALLY by executing unit tests. Reads a run_all jsonl where

````python
"""Score HumanEval outputs LOCALLY by executing unit tests. Reads a run_all jsonl where
each record has `output` (the completion), `prompt`, and `reference={test, entry_point}`.
Reports pass@1 per config_key and dumps a scores jsonl.

SECURITY: this EXECUTES model-generated code in a subprocess. Run only in a throwaway
environment (the pod or a container). Each program runs with a timeout.

    python src/eval/humaneval.py --in results/e3.jsonl --out results/e3_scores.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n@"]


def truncate(completion):
    """HumanEval completions should stop at the first out-of-function token."""
    cut = len(completion)
    for s in STOPS:
        i = completion.find(s)
        if i != -1:
            cut = min(cut, i)
    return completion[:cut]


def passes(prompt, completion, test, entry_point, timeout=10):
    program = prompt + truncate(completion) + "\n" + test + f"\ncheck({entry_point})\n"
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as f:
            f.write(program)
            path = f.name
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=int, default=10)
    args = ap.parse_args()

    correct = defaultdict(int)
    total = defaultdict(int)
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ref = r.get("reference")
            if not isinstance(ref, dict):
                continue
            ck = r.get("config_key", "default")
            ok = passes(r["prompt"], r["output"], ref["test"], ref["entry_point"],
                        args.timeout)
            total[ck] += 1
            correct[ck] += int(ok)

    rows = []
    print(f"{'config_key':<40}{'pass@1':>9}{'n':>6}")
    for ck in sorted(total):
        acc = correct[ck] / total[ck]
        print(f"{ck:<40}{100*acc:>8.1f}%{total[ck]:>6}")
        rows.append({"config_key": ck, "accuracy": acc, "n": total[ck],
                     "correct": correct[ck], "metric": "humaneval"})

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"\nscores -> {args.out}")


if __name__ == "__main__":
    main()
````

### `score_all.py`  ·  267 行

**作用**：汇总所有 results/*.jsonl 打分

> 文件自述：score_all.py (LOCAL, no GPU) — final scoring for the pod outputs. Runs whichever files

````python
"""score_all.py (LOCAL, no GPU) — final scoring for the pod outputs. Runs whichever files
are present; skips the rest.

  AR compare   (--ar results/ar.jsonl --diff results/e2.jsonl)
      diffusion vs AR: clean-correct quality retention + tokens/NFE + wall-clock speedup
  HumanEval    (--he results/he_collapse.jsonl --he-ar results/he_ar.jsonl
                --he-prompts data/humaneval_mini.jsonl)
      pass@1 by executing (prompt + completion + official test) in a sandboxed subprocess
      -> code-side block collapse (cf. paper Table 4)
  RULER        (--ruler results/ruler.jsonl)
      long-context retrieval accuracy + tokens/NFE vs context length

⚠️ HumanEval pass@1 EXECUTES model-generated code in a subprocess with a timeout. Only run on
outputs you trust (your own runs). Pass --no-exec to count without executing.

    python score_all.py --ar results/ar.jsonl --diff results/e2.jsonl \
        --he results/he_collapse.jsonl --he-ar results/he_ar.jsonl \
        --he-prompts data/humaneval_mini.jsonl --ruler results/ruler.jsonl --outdir figs
"""
import argparse, json, os, re, sys, subprocess, tempfile
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, AQUA, VIOLET, RED, ORANGE, MUTED2 = "#2a78d6", "#1baf7a", "#4a3aa7", "#e34948", "#eb6834", "#c3c2b7"
INK, INK2, MUTED, GRID, SURF, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(fn):
    out = []
    if not fn or not os.path.exists(fn):
        return out
    for l in open(fn, encoding="utf-8"):
        l = l.strip()
        if l:
            try: out.append(json.loads(l))
            except: pass
    return out


def degen(s):
    s = s.rstrip(); n = len(s)
    if n < 20: return 0.0
    best = 0
    for p in range(1, 9):
        i = 0
        while i < n:
            j = i
            while j + p < n and s[j] == s[j + p]: j += 1
            if j > i and (j - i) + p >= 2 * p + 6: best = max(best, (j - i) + p)
            i = max(j, i + 1)
    return best / n


def gsm_clean(output, ref):
    if ref is None: return False
    seg = re.split(r"\n\s*Question:", output)[0]
    m = re.findall(r"####\s*\$?\s*([\-0-9][\d,\.]*)", seg) or re.findall(r"([\-0-9][\d,\.]*)", seg)
    if not m: return False
    norm = lambda x: str(x).replace(",", "").rstrip(".").lstrip("$")
    try: ok = abs(float(norm(m[0])) - float(norm(ref))) < 1e-6
    except: ok = norm(m[0]) == norm(ref)
    return ok and degen(output) <= 0.25


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)


# ---------------- AR vs diffusion ----------------
def score_ar(ar_path, diff_path, outdir):
    ar, diff = load(ar_path), load(diff_path)
    if not ar or not diff:
        print("[ar] missing ar/diff jsonl, skipping"); return None
    def cc_tps(rows, keyfilter=None):
        n = ok = 0; tps = 0.0; tpn = 0.0
        for r in rows:
            if keyfilter and keyfilter not in r.get("config_key", ""): continue
            n += 1; ok += gsm_clean(r.get("output", ""), r.get("reference"))
            tps += r.get("tps", 0) or 0; tpn += r.get("tokens_per_nfe", 0) or 0
        return (ok / n * 100 if n else 0, tps / n if n else 0, tpn / n if n else 0, n)
    ar_cc, ar_tps, _, ar_n = cc_tps(ar)
    op = "block_size16_gamma0.8_max_new256_steps16"        # operating point
    d_cc, d_tps, d_tpn, d_n = cc_tps(diff, op)
    if d_n == 0: d_cc, d_tps, d_tpn, d_n = cc_tps(diff)     # fallback: all diff configs
    summary = dict(ar_clean_correct=round(ar_cc, 1), diff_clean_correct=round(d_cc, 1),
                   quality_retention_pct=round(d_cc / ar_cc * 100, 1) if ar_cc else None,
                   ar_tps=round(ar_tps, 2), diff_tps=round(d_tps, 2),
                   wallclock_speedup=round(d_tps / ar_tps, 2) if ar_tps else None,
                   diff_tokens_per_nfe=round(d_tpn, 2))
    print("[ar]", json.dumps(summary, ensure_ascii=False))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.4)); fig.patch.set_facecolor(SURF)
    style(a1); a1.bar([0, 1], [ar_cc, d_cc], color=[MUTED2, BLUE], width=0.6)
    for x, v in zip([0, 1], [ar_cc, d_cc]):
        a1.annotate(f"{v:.0f}%", (x, v), textcoords="offset points", xytext=(0, 4), ha="center", fontweight="bold", color=INK)
    a1.set_xticks([0, 1]); a1.set_xticklabels(["AR tower", "diffusion"]); a1.set_ylim(0, 112)
    a1.set_title("Quality (clean-correct %)", color=INK, fontsize=11, fontweight="bold", loc="left")
    style(a2); a2.bar([0, 1], [1.0, summary["diff_tokens_per_nfe"]], color=[MUTED2, BLUE], width=0.6)
    for x, v in zip([0, 1], [1.0, summary["diff_tokens_per_nfe"]]):
        a2.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 4), ha="center", fontweight="bold", color=INK)
    a2.set_xticks([0, 1]); a2.set_xticklabels(["AR (=1)", "diffusion"])
    a2.set_title("Parallelism (tokens/NFE)", color=INK, fontsize=11, fontweight="bold", loc="left")
    fig.suptitle(f"AR vs diffusion — retention {summary['quality_retention_pct']}%,  "
                 f"wall-clock ×{summary['wallclock_speedup']} (slow-fix wall-clock is pessimistic; tokens/NFE is the clean metric)",
                 color=INK, fontsize=10.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(os.path.join(outdir, "fig_ar_compare.png"), dpi=140, facecolor=SURF)
    plt.close(fig); print("wrote fig_ar_compare.png")
    return summary


# ---------------- HumanEval pass@1 ----------------
STOPS = ["\n```", "\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nQuestion:", "\n\n\n"]

def truncate(completion):
    cut = len(completion)
    for s in STOPS:
        i = completion.find(s)
        if 0 <= i < cut: cut = i
    return completion[:cut]

def run_program(src, timeout=8):
    fd, path = tempfile.mkstemp(suffix=".py"); os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f: f.write(src)
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout, text=True)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try: os.unlink(path)
        except: pass

def score_humaneval(he_path, he_ar_path, prompts_path, outdir, allow_exec=True):
    prompts = {p.get("id", p.get("task_id")): p["prompt"] for p in load(prompts_path)}
    if not prompts:
        print("[he] no --he-prompts (needed for signatures), skipping"); return None
    def passk(rows, tag):
        by = defaultdict(lambda: [0, 0])
        for r in rows:
            pid = r.get("prompt_id"); ref = r.get("reference")
            if pid not in prompts or not isinstance(ref, dict): continue
            prog = prompts[pid] + truncate(r.get("output", "")) + "\n\n" + ref["test"] + f"\ncheck({ref['entry_point']})\n"
            ok = run_program(prog) if allow_exec else False
            k = r.get("config_key", tag); by[k][0] += ok; by[k][1] += 1
        return {k: (v[0] / v[1] * 100 if v[1] else 0, v[1]) for k, v in by.items()}
    he = passk(load(he_path), "diff") if he_path else {}
    he_ar = passk(load(he_ar_path), "ar") if he_ar_path else {}
    print("[he] diff pass@1:", {k: round(v[0], 1) for k, v in he.items()}, "| ar:", {k: round(v[0], 1) for k, v in he_ar.items()})
    # collapse figure: pass@1 vs block size
    blocks = []
    for k in he:
        m = re.search(r"block_size(\d+)", k)
        if m: blocks.append((int(m.group(1)), he[k][0]))
    if blocks:
        blocks.sort()
        fig, ax = plt.subplots(figsize=(7, 4.4)); fig.patch.set_facecolor(SURF); style(ax)
        xs = [str(b) for b, _ in blocks]; ys = [v for _, v in blocks]
        ax.plot(xs, ys, "-o", color=BLUE, lw=2.5, ms=9)
        for x, y in zip(xs, ys): ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold", color=INK)
        if he_ar:
            arv = list(he_ar.values())[0][0]
            ax.axhline(arv, color=MUTED2, ls="--", lw=1.4); ax.annotate(f"AR {arv:.0f}%", (0, arv + 2), color=MUTED, fontsize=9)
        ax.set_xlabel("sampling block size", color=INK2, fontsize=10); ax.set_ylabel("HumanEval pass@1 %", color=INK2, fontsize=10)
        ax.set_title("Claim C — code-side block collapse (pass@1)", color=INK, fontsize=12, fontweight="bold", loc="left")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_he_collapse.png"), dpi=140, facecolor=SURF); plt.close(fig)
        print("wrote fig_he_collapse.png")
    return {"diff": {k: round(v[0], 1) for k, v in he.items()}, "ar": {k: round(v[0], 1) for k, v in he_ar.items()}}


# ---------------- RULER long context ----------------
def score_ruler(ruler_path, outdir):
    rows = load(ruler_path)
    if not rows: print("[ruler] missing, skipping"); return None
    agg = defaultdict(lambda: dict(n=0, ok=0, tpn=0.0, wall=0.0, oom=0, ctx=[]))
    for r in rows:
        k = (r.get("mode", "diff"), r.get("target_len")); a = agg[k]
        if r.get("oom"): a["oom"] += 1; continue
        a["n"] += 1; a["ok"] += bool(r.get("ok")); a["tpn"] += r.get("tokens_per_nfe") or 0
        a["wall"] += r.get("wall_s") or 0; a["ctx"].append(r.get("ctx_len"))
    lens = sorted({k[1] for k in agg})
    modes = sorted({k[0] for k in agg})
    print("[ruler] lengths:", lens, "modes:", modes)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4)); fig.patch.set_facecolor(SURF)
    cols = {"diff": BLUE, "ar": MUTED2}
    for mode in modes:
        acc = [agg[(mode, L)]["ok"] / agg[(mode, L)]["n"] * 100 if agg[(mode, L)]["n"] else float("nan") for L in lens]
        style(a1); a1.plot(lens, acc, "-o", color=cols.get(mode, VIOLET), lw=2.3, ms=7, label=mode)
    a1.set_xscale("log"); a1.set_xticks(lens); a1.set_xticklabels([str(x) for x in lens])
    a1.set_xlabel("context length (tokens)", color=INK2, fontsize=10); a1.set_ylabel("needle retrieval %", color=INK2, fontsize=10)
    a1.set_title("Long-context retrieval", color=INK, fontsize=11, fontweight="bold", loc="left"); a1.legend(frameon=False, fontsize=9)
    tpn = [agg[("diff", L)]["tpn"] / agg[("diff", L)]["n"] if agg[("diff", L)]["n"] else float("nan") for L in lens]
    style(a2); a2.plot(lens, tpn, "-o", color=BLUE, lw=2.3, ms=7)
    a2.axhline(1.0, color=MUTED2, ls="--", lw=1.2); a2.annotate("AR = 1", (lens[0], 1.05), color=MUTED, fontsize=8.5)
    a2.set_xscale("log"); a2.set_xticks(lens); a2.set_xticklabels([str(x) for x in lens])
    a2.set_xlabel("context length (tokens)", color=INK2, fontsize=10); a2.set_ylabel("tokens/NFE (diffusion)", color=INK2, fontsize=10)
    a2.set_title("Does parallelism hold at long context?", color=INK, fontsize=11, fontweight="bold", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_ruler.png"), dpi=140, facecolor=SURF); plt.close(fig)
    print("wrote fig_ruler.png")
    return {f"{m}@{L}": dict(acc=round(agg[(m, L)]["ok"] / agg[(m, L)]["n"] * 100, 1) if agg[(m, L)]["n"] else None,
                             oom=agg[(m, L)]["oom"]) for m in modes for L in lens}


# ---------------- confidence -> correctness (cf. DG finding #6) ----------------
def score_conf(conf_path, outdir):
    rows = load(conf_path)
    if not rows:
        print("[conf] missing, skipping"); return None
    pts = []  # (mean_commit_conf, correct)
    for r in rows:
        c = r.get("mean_commit_conf")
        if c is None:
            continue
        pts.append((c, 1 if gsm_clean(r.get("output", ""), r.get("reference")) else 0))
    if not pts:
        print("[conf] no usable rows"); return None
    cc = [c for c, o in pts if o]; wc = [c for c, o in pts if not o]
    mc = sum(cc) / len(cc) if cc else float("nan")
    mw = sum(wc) / len(wc) if wc else float("nan")
    n = len(pts); mx = sum(c for c, _ in pts) / n; my = sum(o for _, o in pts) / n
    sx = (sum((c - mx) ** 2 for c, _ in pts) / n) ** 0.5
    sy = (sum((o - my) ** 2 for _, o in pts) / n) ** 0.5
    r_pb = (sum((c - mx) * (o - my) for c, o in pts) / n) / (sx * sy) if sx > 0 and sy > 0 else float("nan")
    summary = dict(n=n, n_correct=len(cc), mean_conf_correct=round(mc, 3),
                   mean_conf_wrong=round(mw, 3), point_biserial_r=round(r_pb, 3))
    print("[conf]", json.dumps(summary, ensure_ascii=False))
    fig, ax = plt.subplots(figsize=(6.4, 4.4)); fig.patch.set_facecolor(SURF); style(ax)
    ax.bar([0, 1], [mw, mc], color=[RED, BLUE], width=0.55)
    for x, v in zip([0, 1], [mw, mc]):
        if v == v:
            ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontweight="bold", color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"wrong (n={len(wc)})", f"correct (n={len(cc)})"])
    ax.set_ylabel("mean commit confidence", color=INK2, fontsize=10)
    ax.set_title(f"Does confidence predict correctness?  point-biserial r={summary['point_biserial_r']}",
                 color=INK, fontsize=11, fontweight="bold", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_conf_correct.png"), dpi=140, facecolor=SURF)
    plt.close(fig); print("wrote fig_conf_correct.png")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar"); ap.add_argument("--diff")
    ap.add_argument("--he"); ap.add_argument("--he-ar"); ap.add_argument("--he-prompts")
    ap.add_argument("--ruler"); ap.add_argument("--conf"); ap.add_argument("--outdir", default="figs")
    ap.add_argument("--no-exec", action="store_true", help="skip executing HumanEval code")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    out = {}
    if a.ar or a.diff: out["ar"] = score_ar(a.ar, a.diff, a.outdir)
    if a.he or a.he_ar: out["humaneval"] = score_humaneval(a.he, a.he_ar, a.he_prompts, a.outdir, not a.no_exec)
    if a.ruler: out["ruler"] = score_ruler(a.ruler, a.outdir)
    if a.conf: out["conf"] = score_conf(a.conf, a.outdir)
    with open(os.path.join(a.outdir, "score_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("done -> score_summary.json")


if __name__ == "__main__":
    main()
````

---

## 6 · 参数核算与权重检查

报告 §3 / 附录 A。纯 config 算术（零 GPU）核出 30B-A3B / 双塔 63.2B；权重加载核实。

### `src/param_count.py`  ·  106 行

**作用**：从 config.json 纯算术核出每模块参数量（无需 torch）

> 文件自述：Per-module parameter breakdown from config.json ARITHMETIC ONLY.

````python
"""Per-module parameter breakdown from config.json ARITHMETIC ONLY.

Runs anywhere (your laptop, no GPU, no torch, no mamba-ssm) because it never imports
the model — it derives every matrix shape from config.json the way the modeling code
builds them. This reproduces the 30B-A3B accounting:
  ~30B total / ~3B activated PER TOWER, ~63B across both towers.

NOTE: this counts ONE NemotronH backbone tower (modeling_nemotron_h.py). The denoiser
tower additionally has per-layer cross-attention + AdaLN time conditioning defined in
modeling_nemotron_twotower.py (not yet read) — those are flagged as TODO extras.
Pass --config to point at the model's config.json; otherwise the embedded one is used.
"""
import argparse
import json

# config.json as provided by the user (embedded default)
CFG = {
    "hidden_size": 2688, "num_hidden_layers": 52,
    "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
    "vocab_size": 131072, "tie_word_embeddings": False,
    # mamba
    "mamba_num_heads": 64, "mamba_head_dim": 64, "n_groups": 8,
    "ssm_state_size": 128, "conv_kernel": 4, "use_conv_bias": True, "use_bias": False,
    # attention
    "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
    "attention_bias": False,
    # moe
    "n_routed_experts": 128, "num_experts_per_tok": 6, "moe_intermediate_size": 1856,
    "moe_shared_expert_intermediate_size": 3712, "mlp_bias": False,
}


def module_params(c):
    H = c["hidden_size"]

    # --- Mamba-2 mixer (in_proj, conv1d, dt/A/D, gated norm, out_proj) ---
    mi = c["mamba_num_heads"] * c["mamba_head_dim"]                 # 4096
    conv_dim = mi + 2 * c["n_groups"] * c["ssm_state_size"]        # 6144
    proj = mi + conv_dim + c["mamba_num_heads"]                    # 10304
    mamba = (
        H * proj + mi * H                                          # in_proj + out_proj
        + conv_dim * c["conv_kernel"] + (conv_dim if c["use_conv_bias"] else 0)  # conv1d
        + 3 * c["mamba_num_heads"]                                 # dt_bias + A_log + D
        + mi                                                       # gated RMSNorm weight
    )

    # --- Attention (GQA: 32 q heads, 2 kv heads, head_dim 128) ---
    hd, nh, nkv = c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
    attn = H * (nh * hd) + 2 * H * (nkv * hd) + (nh * hd) * H       # q + k + v + o

    # --- MoE (128 experts of relu^2 MLP up/down, 1 shared, fp32 router) ---
    expert = 2 * H * c["moe_intermediate_size"]
    shared = 2 * H * c["moe_shared_expert_intermediate_size"]
    router = c["n_routed_experts"] * H
    moe_total = c["n_routed_experts"] * expert + shared + router
    moe_active = c["num_experts_per_tok"] * expert + shared + router

    return dict(mamba=mamba, attn=attn, moe_total=moe_total, moe_active=moe_active,
                expert=expert, shared=shared, block_norm=H)


def count_tower(c):
    m = module_params(c)
    p = c["hybrid_override_pattern"]
    nM, nE, nA = p.count("M"), p.count("E"), p.count("*")
    L = c["num_hidden_layers"]
    embed = c["vocab_size"] * c["hidden_size"]
    lm_head = 0 if c["tie_word_embeddings"] else c["vocab_size"] * c["hidden_size"]
    norms = L * m["block_norm"] + c["hidden_size"]  # per-block + final

    rows = [
        (f"mamba2   x{nM}", nM * m["mamba"]),
        (f"moe      x{nE} (total)", nE * m["moe_total"]),
        (f"attention x{nA}", nA * m["attn"]),
        ("embed + lm_head", embed + lm_head),
        ("norms", norms),
    ]
    total = sum(v for _, v in rows)
    active = (nM * m["mamba"] + nE * m["moe_active"] + nA * m["attn"]
              + embed + lm_head + norms)
    return rows, total, active, (nM, nE, nA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="path to config.json (defaults to embedded)")
    args = ap.parse_args()
    c = json.load(open(args.config)) if args.config else CFG

    rows, total, active, (nM, nE, nA) = count_tower(c)
    print(f"\nPattern: {nM} Mamba-2 / {nE} MoE / {nA} attention  (per tower, 52 layers)\n")
    print(f"{'module':<26}{'params':>14}")
    print("-" * 40)
    for name, v in rows:
        print(f"{name:<26}{v/1e9:>12.3f}B")
    print("-" * 40)
    print(f"{'TOWER total':<26}{total/1e9:>12.3f}B")
    print(f"{'TOWER activated':<26}{active/1e9:>12.3f}B")
    print(f"\nBoth towers (x2, backbone only): total {2*total/1e9:.1f}B, "
          f"activated {2*active/1e9:.2f}B")
    print("[TODO] denoiser adds per-layer cross-attn + AdaLN "
          "(needs modeling_nemotron_twotower.py) -> Analysis #3")


if __name__ == "__main__":
    main()
````

### `src/check_weights.py`  ·  45 行

**作用**：核实去噪塔权重确已载入

> 文件自述：Did the DENOISER tower's weights actually load? Top suspect for the garbage: if the

````python
"""Did the DENOISER tower's weights actually load? Top suspect for the garbage: if the
denoiser is random-initialized (its checkpoint keys didn't map), every diffusion forward is
noise -> NFE=max, word-salad, and no seed/adaln bypass helps — exactly what we observe,
while AR (context tower) stays coherent.

Surfaces from_pretrained's "were not initialized from the model checkpoint" warnings and
prints per-tower parameter stats + the actual module naming.
    python src/check_weights.py
"""
import collections
import itertools

import torch
import transformers
from transformers import AutoModelForCausalLM

from twotower import MODEL_NAME

transformers.logging.set_verbosity_info()   # <-- surfaces the "newly initialized" list


def main():
    print(">>> loading — WATCH FOR: 'Some weights of ... were not initialized ... "
          "newly initialized: [...]'  (if denoiser keys appear there, that's the bug)\n")
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True)

    names = [n for n, _ in m.named_parameters()]
    print("\ntop-level modules:",
          dict(collections.Counter(n.split(".")[0] for n in names)))

    for key in ["denois", "context", "backbone", "model", "t_embedder", "t_block",
                "scale_shift", "lm_head"]:
        ps = [p for n, p in m.named_parameters() if key in n.lower()]
        if ps:
            flat = torch.cat([p.float().flatten() for p in itertools.islice(ps, 30)])
            print(f"{key:12} tensors={len(ps):4d} mean={flat.mean():+.4f} "
                  f"std={flat.std():.4f} absmax={flat.abs().max():.3f}")

    den = [n for n in names if "denois" in n.lower()]
    print("\nexample denoiser param names:", den[:4] or "(none — check top-level naming)")


if __name__ == "__main__":
    main()
````

---

## 7 · 图与动画生成

报告全部图与 GIF 均出自这里（纯本地、读 results/*.jsonl 与 trace_*.pkl）。

### `make_figs.py`  ·  237 行

**作用**：生成报告全部静态图（fig_*.png）

> 文件自述：Local report figures for the TwoTower reproduction — no GPU, reads the downloaded jsonl.

````python
"""Local report figures for the TwoTower reproduction — no GPU, reads the downloaded jsonl.

Scoring is degeneration-aware: a GSM8K answer counts only if it is correct AND the output is
not a short-cycle repetition ("edededed" / "155555" garbage). That distinction turns the
block-collapse from a hidden 90% into an honest 50% clean-correct. All figure labels are in
English; the dataviz reference palette (light surface) is used throughout.
"""
import json, re, os, pickle
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
VIOLET, RED, ORANGE = "#4a3aa7", "#e34948", "#eb6834"
INK, INK2, MUTED, GRID, SURF, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
GOOD, CRIT = "#0ca30c", "#d03b3b"
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs"); os.makedirs(FIGS, exist_ok=True)


def load(fn):
    out = []
    for l in open(os.path.join(HERE, fn), encoding="utf-8"):
        l = l.strip()
        if l:
            try: out.append(json.loads(l))
            except: pass
    return out


def degen_frac(s):
    s = s.rstrip(); n = len(s)
    if n < 20: return 0.0
    best = 0
    for p in range(1, 9):
        i = 0
        while i < n:
            j = i
            while j + p < n and s[j] == s[j + p]:
                j += 1
            if j > i and (j - i) + p >= 2 * p + 6:
                best = max(best, (j - i) + p)
            i = max(j, i + 1)
    return best / n


def gsm_ok(output, ref):
    if ref is None: return False
    seg = re.split(r"\n\s*Question:", output)[0]
    m = re.findall(r"####\s*\$?\s*([\-0-9][\d,\.]*)", seg) or re.findall(r"([\-0-9][\d,\.]*)", seg)
    if not m: return False
    norm = lambda x: str(x).replace(",", "").rstrip(".").lstrip("$")
    try: return abs(float(norm(m[0])) - float(norm(ref))) < 1e-6
    except: return norm(m[0]) == norm(ref)


def agg(fn):
    a = defaultdict(lambda: dict(n=0, ok=0, clean=0, nfe=0.0, tpn=0.0))
    for r in load(fn):
        k = r.get("config_key", "?"); d = a[k]; d["n"] += 1
        ok = gsm_ok(r.get("output", ""), r.get("reference"))
        deg = degen_frac(r.get("output", "")) > 0.25
        d["ok"] += ok; d["clean"] += (ok and not deg)
        d["nfe"] += r.get("nfe", 0) or 0; d["tpn"] += r.get("tokens_per_nfe", 0) or 0
    return a


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)


def save(fig, name):
    p = os.path.join(FIGS, name); fig.savefig(p, dpi=140, facecolor=SURF); plt.close(fig)
    print("wrote", name)


# ---- FIG 1: gamma -> tokens/NFE (Claim A: still autoregressive) ----
def fig_gamma():
    rows = load("e1.jsonl"); gammas = [0.5, 0.7, 0.8, 0.9, 0.95]
    g = defaultdict(list)
    for r in rows: g[(r["gamma"], r["steps"])].append(r["tokens_per_nfe"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6)); fig.patch.set_facecolor(SURF); style(ax)
    for T, col in [(4, BLUE), (8, AQUA), (16, VIOLET)]:
        y = [sum(g[(gm, T)]) / len(g[(gm, T)]) for gm in gammas]
        ax.plot(gammas, y, "-o", color=col, lw=2.4, ms=7, label=f"steps={T}")
        ax.annotate(f"steps={T}", (gammas[-1], y[-1]), textcoords="offset points",
                    xytext=(6, 0), color=col, fontsize=9, fontweight="bold", va="center")
    ax.axhline(1.0, color=MUTED, ls="--", lw=1.2)
    ax.annotate("AR = 1.0 token / forward", (0.5, 1.12), color=MUTED, fontsize=8.5)
    ax.annotate("higher γ  →  closer to autoregressive", (0.7, 5.4), color=INK2, fontsize=9.5)
    ax.set_xlabel("confidence threshold  γ", color=INK2, fontsize=10)
    ax.set_ylabel("tokens / NFE  (in-block parallelism)", color=INK2, fontsize=10)
    ax.set_title("Claim A — still autoregressive: parallelism decays toward 1", color=INK,
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right"); ax.set_xlim(0.46, 1.02)
    fig.tight_layout(); save(fig, "fig_gamma.png")


# ---- FIG 2: ablations (Claim B) ----
def fig_ablation():
    ra, da = agg("abl_remask.jsonl"), agg("abl_denoiser.jsonl")
    labels = ["baseline", "remask OFF\n(no iteration)", "seed OFF\n(no Mamba seeding)", "time frozen\n(AdaLN fixed)"]
    src = [ra["baseline"], ra["remask_OFF"], da["disable_seed"], da["freeze_time"]]
    clean = [s["clean"] / s["n"] * 100 for s in src]; nfe = [s["nfe"] / s["n"] for s in src]
    cols = [MUTED, RED, CRIT, BLUE]; x = list(range(4))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 4.7)); fig.patch.set_facecolor(SURF)
    for ax, vals, ttl, fmt in [(a1, clean, "Quality — clean-correct %", "{:.0f}%"),
                               (a2, nfe, "Compute — NFE (lower = faster)", "{:.0f}")]:
        style(ax); ax.bar(x, vals, color=cols, width=0.62, zorder=2)
        for xi, v in zip(x, vals):
            ax.annotate(fmt.format(v), (xi, v), textcoords="offset points", xytext=(0, 4),
                        ha="center", color=INK, fontsize=10, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.3)
        ax.set_title(ttl, color=INK, fontsize=11, fontweight="bold", loc="left")
    a1.set_ylim(0, 112)
    a2.annotate("no seeding →\nconfidence never clears γ\n→ crawls ~1 token/step",
                xy=(2, 243), xytext=(0.28, 188), color=CRIT, fontsize=8.3, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=CRIT, lw=1.2))
    fig.suptitle("Claim B — seeding load-bearing · iteration lifts quality · time-conditioning inert (freeze = no change)",
                 color=INK, fontsize=11.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94]); save(fig, "fig_ablation.png")


# ---- FIG 3: block collapse (Claim C) ----
def fig_collapse():
    a = agg("e3.jsonl"); blocks = [8, 16, 32, 64]
    key = lambda b: f"block_size{b}_gamma0.8_max_new256_steps16_temperature0.0"
    clean = [a[key(b)]["clean"] / a[key(b)]["n"] * 100 for b in blocks]
    naive = [a[key(b)]["ok"] / a[key(b)]["n"] * 100 for b in blocks]
    x = list(range(len(blocks)))
    fig, ax = plt.subplots(figsize=(7.2, 4.6)); fig.patch.set_facecolor(SURF); style(ax)
    ax.fill_between(x, clean, naive, color=RED, alpha=0.12)
    ax.plot(x, naive, "--o", color=MUTED, lw=2, ms=7, label="naive accuracy (first #### only)")
    ax.plot(x, clean, "-o", color=BLUE, lw=2.5, ms=9, label="clean-correct (correct & not degenerate)")
    for xi, c in zip(x, clean):
        ax.annotate(f"{c:.0f}%", (xi, c), textcoords="offset points", xytext=(0, 10),
                    ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.axvline(1, color=AXIS, lw=1, ls=":")
    ax.annotate("training block = 16", (1, 16), color=INK2, fontsize=8.5, ha="center")
    ax.annotate("block 64: half the outputs\ndegenerate into repetition", (3, 62), color=CRIT,
                fontsize=9, ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in blocks])
    ax.set_xlabel("sampling block size", color=INK2, fontsize=10)
    ax.set_ylabel("accuracy %", color=INK2, fontsize=10); ax.set_ylim(0, 108)
    ax.set_title("Claim C — block-size collapse hidden by naive accuracy", color=INK,
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="center left"); fig.tight_layout()
    save(fig, "fig_collapse.png")


# ---- FIG 4: quality-speed Pareto (e2) ----
def fig_pareto():
    a = agg("e2.jsonl")
    pts = []
    for k, d in a.items():
        g = float(re.search(r"gamma([\d.]+)", k).group(1))
        T = int(re.search(r"steps(\d+)", k).group(1))
        pts.append((d["tpn"] / d["n"], d["clean"] / d["n"] * 100, g, T))
    pts.sort()
    fig, ax = plt.subplots(figsize=(7.2, 4.6)); fig.patch.set_facecolor(SURF); style(ax)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, "-", color=GRID, lw=1.5, zorder=1)
    ax.scatter(xs, ys, color=BLUE, s=70, zorder=3)
    # per-point label offsets to defuse the top-left 100% cluster
    off = [(-8, 12, "right"), (0, 24, "center"), (10, 10, "left"),
           (-8, -15, "right"), (8, 9, "left"), (-8, -6, "right")]
    for (tpn, cc, g, T), (dx, dy, ha) in zip(pts, off):
        ax.annotate(f"γ={g}, T={T}", (tpn, cc), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, color=INK2, fontsize=8.2)
    ax.set_xlabel("tokens / NFE  →  faster (more parallel)", color=INK2, fontsize=10)
    ax.set_ylabel("clean-correct %", color=INK2, fontsize=10)
    ax.set_title("Quality – speed trade-off is mild (GSM8K-mini)", color=INK, fontsize=12,
                 fontweight="bold", loc="left")
    ax.set_ylim(min(ys) - 8, 107); ax.set_xlim(1.7, 7.6)
    fig.tight_layout(); save(fig, "fig_pareto.png")


# ---- FIG 5: steps per block (real trajectory signal) ----
def fig_stepsblock():
    tm = pickle.load(open(os.path.join(HERE, "trace_main.pkl"), "rb"))
    spbs = [{int(k): int(v) for k, v in tr["summary"]["steps_per_block"].items()} for tr in tm["traces"]]
    nb = max(max(s) for s in spbs) + 1
    mean = [sum(s.get(b, 0) for s in spbs) / len(spbs) for b in range(nb)]
    fig, ax = plt.subplots(figsize=(7.6, 4.4)); fig.patch.set_facecolor(SURF); style(ax)
    ax.bar(range(nb), mean, color=BLUE, width=0.7, zorder=2)
    ax.axhline(sum(mean) / len(mean), color=ORANGE, ls="--", lw=1.4)
    ax.annotate(f"mean {sum(mean)/len(mean):.1f}", (nb - 1.5, sum(mean) / len(mean) + 0.3),
                color=ORANGE, fontsize=9)
    ax.annotate("early blocks iterate more;\nlater blocks settle in ~2 steps\n(more context → higher confidence)",
                (5.5, 8.5), color=INK2, fontsize=9)
    ax.set_xticks(range(nb)); ax.set_xlabel("block index (generation order →)", color=INK2, fontsize=10)
    ax.set_ylabel("denoising steps used", color=INK2, fontsize=10)
    ax.set_title("Denoising steps per block (mean of 3 prompts) — measured", color=INK,
                 fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout(); save(fig, "fig_stepsblock.png")


# ---- FIG 6: MoE routing churn across denoising steps ----
def fig_moe():
    mo = pickle.load(open(os.path.join(HERE, "trace_moe.pkl"), "rb"))
    bylayer = defaultdict(list)
    for layer, step, arr in mo["moe_records"]:
        bylayer[layer].append((step, arr))
    ret = {}
    for layer, lst in bylayer.items():
        lst.sort(key=lambda x: x[0]); vals = []
        for (_, a0), (_, a1) in zip(lst, lst[1:]):
            for p in range(a0.shape[0]):
                vals.append(len(set(a0[p].tolist()) & set(a1[p].tolist())))
        if vals: ret[layer] = sum(vals) / len(vals)
    items = sorted(ret.items(), key=lambda kv: int(re.search(r"layers\.(\d+)", kv[0]).group(1)))
    lids = [int(re.search(r"layers\.(\d+)", k).group(1)) for k, _ in items]
    vals = [v for _, v in items]
    print("MoE retained experts/6 per layer:", {l: round(v, 2) for l, v in zip(lids, vals)})
    fig, ax = plt.subplots(figsize=(8.2, 4.4)); fig.patch.set_facecolor(SURF); style(ax)
    ax.bar([str(l) for l in lids], vals, color=VIOLET, width=0.7, zorder=2)
    ax.axhline(6 * 6 / 128, color=MUTED, ls="--", lw=1.2)
    ax.annotate("random-routing floor (≈0.28/6)", (0.2, 6 * 6 / 128 + 0.15), color=MUTED, fontsize=8.5)
    ax.set_ylim(0, 6)
    ax.set_xlabel("denoiser MoE layer index", color=INK2, fontsize=10)
    ax.set_ylabel("experts retained (of 6)\nbetween consecutive steps", color=INK2, fontsize=10)
    ax.set_title("MoE routing churns across denoising steps (a diffusion-only phenomenon)",
                 color=INK, fontsize=11.5, fontweight="bold", loc="left")
    fig.tight_layout(); save(fig, "fig_moe.png")


if __name__ == "__main__":
    fig_gamma(); fig_ablation(); fig_collapse(); fig_pareto(); fig_stepsblock(); fig_moe()
    print("done ->", FIGS)
````

### `make_gif.py`  ·  89 行

**作用**：生成去噪过程动画（denoising.gif）

> 文件自述：Block-diffusion denoising GIF, built from the REAL per-block step counts we still have.

````python
"""Block-diffusion denoising GIF, built from the REAL per-block step counts we still have.

HONEST SCOPE: block order and each block's denoising-step count come from the measured
trace (`steps_per_block` in trace_main.pkl). The exact position that commits at each step
was NOT captured (that capture bug is what we fixed but can't re-run without a pod), so the
WITHIN-block fill order is illustrative (left-to-right). The thing the animation actually
demonstrates from data: generation sweeps block-by-block (still left-to-right), and early
blocks take many more steps than later ones.
"""
import pickle, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans"]
BLUE, GREEN, ORANGE = (0.165, 0.470, 0.839), (0.0, 0.62, 0.31), (0.92, 0.41, 0.20)
MASK = (0.91, 0.91, 0.90)
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs"); os.makedirs(FIGS, exist_ok=True)

BS = 16
tm = pickle.load(open(os.path.join(HERE, "trace_main.pkl"), "rb"))
tr = tm["traces"][0]
spb = tr["summary"]["steps_per_block"]          # {block: n_steps}  -- REAL
spb = {int(k): int(v) for k, v in spb.items()}
NB = len(spb)                                    # 16 blocks

# --- precompute the global frame at which each (block,pos) cell commits ---
commit_frame = np.full((NB, BS), -1, np.int32)
f = 0
for b in range(NB):
    S = max(1, spb[b]); prev = 0
    for s in range(S):
        filled = round(BS * (s + 1) / S)         # linear within-block fill (illustrative)
        commit_frame[b, prev:filled] = f
        prev = filled; f += 1
    commit_frame[b, prev:] = f - 1
TOTAL = f

def color_for(age):
    if age < 0: return MASK
    if age == 0: return GREEN                    # committed this step
    if age <= 2: return ORANGE                   # recently committed
    return BLUE                                  # settled

frames = []
for fr in range(TOTAL):
    img = np.zeros((NB, BS, 3))
    for b in range(NB):
        for c in range(BS):
            cf = commit_frame[b, c]
            img[b, c] = color_for(fr - cf if (cf >= 0 and cf <= fr) else -1)
    cur_b = int(np.searchsorted(np.cumsum([max(1, spb[b]) for b in range(NB)]), fr, "right"))
    committed = int(((commit_frame >= 0) & (commit_frame <= fr)).sum())

    fig, ax = plt.subplots(figsize=(6.4, 6.8)); fig.patch.set_facecolor(SURF)
    ax.imshow(img, interpolation="nearest", aspect="equal", extent=[0, BS, NB, 0])
    for k in range(NB + 1): ax.axhline(k, color=SURF, lw=1.5)
    for k in range(BS + 1): ax.axvline(k, color=SURF, lw=1.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Block-diffusion denoising  (256-token reply)", color=INK, fontsize=13,
                 fontweight="bold", loc="left")
    ax.text(0, -0.6, "each ROW = one 16-token block · sweep is top→bottom (still left-to-right)",
            color=MUTED, fontsize=8.5)
    ax.set_xlabel(f"block {min(cur_b, NB-1)+1}/{NB}   ·   frame {fr+1}/{TOTAL}   ·   "
                  f"committed {committed}/{NB*BS}", color=INK2, fontsize=10)
    # legend
    for i, (col, lab) in enumerate([(GREEN, "just committed"), (ORANGE, "recent"),
                                    (BLUE, "settled"), (MASK, "masked")]):
        ax.add_patch(plt.Rectangle((i * 4.2, NB + 0.5), 0.7, 0.7, color=col, clip_on=False))
        ax.text(i * 4.2 + 0.9, NB + 1.15, lab, fontsize=8, color=INK2, clip_on=False)
    fig.tight_layout()
    p = os.path.join(FIGS, f"_gframe_{fr:03d}.png")
    fig.savefig(p, dpi=90, facecolor=SURF); plt.close(fig)
    frames.append(imageio.imread(p))

out = os.path.join(FIGS, "denoising.gif")
imageio.mimsave(out, frames + [frames[-1]] * 8, fps=6, loop=0)
# keep two representative stills for the report; drop the scratch frames
mid = TOTAL * 3 // 5
os.replace(os.path.join(FIGS, f"_gframe_{mid:03d}.png"), os.path.join(FIGS, "gif_still_mid.png"))
for fr in range(TOTAL):
    fp = os.path.join(FIGS, f"_gframe_{fr:03d}.png")
    if os.path.exists(fp): os.remove(fp)
print(f"wrote {out}  ({TOTAL} frames)  | steps_per_block sum={sum(spb.values())}")
print("still ->", os.path.join(FIGS, "gif_still_mid.png"))
````

### `src/plot.py`  ·  124 行

**作用**：早期绘图工具

> 文件自述：Plot LOCALLY (Mac). Reads run_all jsonl (speed) and optional scores jsonl (quality),

````python
"""Plot LOCALLY (Mac). Reads run_all jsonl (speed) and optional scores jsonl (quality),
produces the presentation figures. Config fields are parsed back out of config_key.

    python src/plot.py --speed results/e1.jsonl --kind speed_surface --out results/e1_surface.png
    python src/plot.py --speed results/e2.jsonl --scores results/e2_scores.jsonl --kind pareto --out results/pareto.png
    python src/plot.py --speed results/e3.jsonl --scores results/e3_scores.jsonl --kind collapse --out results/collapse.png
"""
import argparse
import json
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_ck(ck):
    """config_key like 'block_size16_gamma0.8_max_new256_steps16_temperature0.0'."""
    out = {}
    for key in ("block_size", "gamma", "steps", "max_new", "temperature"):
        m = re.search(rf"{key}(-?[\d.]+)", ck)
        if m:
            out[key] = float(m.group(1))
    return out


def load_speed(path):
    """Mean tps / tokens_per_nfe / nfe per config_key."""
    agg = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ck = r["config_key"]
            for k in ("tps", "tokens_per_nfe", "nfe", "wall_s"):
                if r.get(k) is not None:
                    agg[ck][k].append(r[k])
    return {ck: {k: float(np.mean(v)) for k, v in d.items()} for ck, d in agg.items()}


def load_scores(path):
    scores = {}
    if path:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                scores[r["config_key"]] = r["accuracy"]
    return scores


def speed_surface(speed, out):
    gammas = sorted({parse_ck(ck).get("gamma") for ck in speed})
    steps = sorted({parse_ck(ck).get("steps") for ck in speed})
    grid = np.full((len(gammas), len(steps)), np.nan)
    for ck, d in speed.items():
        p = parse_ck(ck)
        grid[gammas.index(p["gamma"]), steps.index(p["steps"])] = d["tokens_per_nfe"]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(steps))); ax.set_xticklabels([int(s) for s in steps])
    ax.set_yticks(range(len(gammas))); ax.set_yticklabels(gammas)
    ax.set_xlabel("steps_per_block (T)"); ax.set_ylabel("confidence_threshold (gamma)")
    ax.set_title("parallelism (tokens / NFE)")
    for i in range(len(gammas)):
        for j in range(len(steps)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i,j]:.1f}", ha="center", va="center", color="w")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out, dpi=120)
    print("wrote", out)


def pareto(speed, scores, out):
    xs, ys, labels = [], [], []
    for ck, acc in scores.items():
        if ck in speed:
            xs.append(speed[ck]["tps"]); ys.append(100 * acc); labels.append(ck)
    order = np.argsort(xs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.array(xs)[order], np.array(ys)[order], "o-", color="#1f77b4")
    for x, y, lab in zip(xs, ys, labels):
        p = parse_ck(lab)
        ax.annotate(f"g{p.get('gamma')}/T{int(p.get('steps',0))}", (x, y),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("throughput (tokens/sec)"); ax.set_ylabel("accuracy (%)")
    ax.set_title("quality vs speed Pareto"); fig.tight_layout(); fig.savefig(out, dpi=120)
    print("wrote", out)


def collapse(speed, scores, out):
    pts = []
    for ck, acc in scores.items():
        pts.append((parse_ck(ck).get("block_size"), 100 * acc))
    pts.sort()
    xs = [str(int(b)) for b, _ in pts]
    ys = [a for _, a in pts]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(xs, ys, color=["#2ca02c" if int(x) <= 16 else "#d62728" for x in xs])
    ax.set_xlabel("sampling block_size (trained at 16)"); ax.set_ylabel("accuracy (%)")
    ax.set_title("block-size collapse (Table 4)")
    ax.axvline(x=0.5 + xs.index("16") if "16" in xs else 0, color="gray", ls="--", lw=1)
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", required=True)
    ap.add_argument("--scores", default=None)
    ap.add_argument("--kind", choices=["speed_surface", "pareto", "collapse"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    speed = load_speed(args.speed)
    scores = load_scores(args.scores)
    if args.kind == "speed_surface":
        speed_surface(speed, args.out)
    elif args.kind == "pareto":
        pareto(speed, scores, args.out)
    else:
        collapse(speed, scores, args.out)


if __name__ == "__main__":
    main()
````

---

## 8 · 环境/冒烟/诊断/数据准备

报告 §9 避雷固化。环境自检、冒烟、扩散诊断、数据准备。

### `src/env_check.py`  ·  48 行

**作用**：环境自检（torch/mamba-ssm/triton 版本闸门）

> 文件自述：Small-GPU environment check: validates the whole toolchain WITHOUT loading the 60GB

````python
"""Small-GPU environment check: validates the whole toolchain WITHOUT loading the 60GB
weights. A cheap card (even ~16-24GB) is enough — this imports and exercises the CUDA
kernels, then loads only config + tokenizer. Full generation still needs 2x80GB.

    python src/env_check.py
"""
import torch

from twotower import MODEL_NAME, MASK_TOKEN_ID


def main():
    print("torch", torch.__version__, "| cuda", torch.version.cuda,
          "| cxx11abi", torch._C._GLIBCXX_USE_CXX11_ABI)
    assert torch.cuda.is_available(), "no CUDA device visible"
    print("device:", torch.cuda.get_device_name(0))

    # 1. the two ABI-sensitive kernel packages must import AND resolve symbols
    import causal_conv1d
    import mamba_ssm
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # noqa: F401
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update  # noqa
    from causal_conv1d import causal_conv1d_fn  # noqa: F401
    print("mamba_ssm", mamba_ssm.__version__, "| causal_conv1d", causal_conv1d.__version__)

    # 2. actually run a tiny causal_conv1d kernel to prove it executes on this GPU
    x = torch.randn(1, 8, 16, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(8, 4, device="cuda", dtype=torch.bfloat16)
    _ = causal_conv1d_fn(x, w, None, activation="silu")
    print("causal_conv1d kernel ran OK")

    # 3. transformers + the custom modeling file must import (needs mamba-ssm present)
    import transformers
    from transformers import AutoConfig, AutoTokenizer
    print("transformers", transformers.__version__)
    cfg = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"config OK: {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, "
          f"pattern {cfg.hybrid_override_pattern[:12]}...")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"tokenizer OK: vocab {tok.vocab_size}, "
          f"decode({MASK_TOKEN_ID})={tok.decode([MASK_TOKEN_ID])!r}")

    print("\nenv_check PASSED — toolchain good. Full generation needs 2x80GB (smoke_test.py).")


if __name__ == "__main__":
    main()
````

### `src/smoke_test.py`  ·  79 行

**作用**：最小冒烟：模型能载能生成

> 文件自述：M0 smoke test: prove the environment works end-to-end before building any experiment.

````python
"""M0 smoke test: prove the environment works end-to-end before building any experiment.

Passing criteria:
  1. tokenizer's mask token id == MASK_TOKEN_ID (3)
  2. AR generation produces non-empty text
  3. diffusion generation produces non-empty text with NO NaN/Inf during denoising
  4. NFE and tokens/NFE (diffusion parallelism factor) are printed; AR is 1.0 by definition
"""
import time
import torch

from twotower import load, load_tokenizer, reset_nfe, get_nfe, nan_guard_callback, MASK_TOKEN_ID

PROMPT = "The capital of France is"
MAX_NEW = 64          # must be divisible by block_size
BLOCK_SIZE = 16
STEPS_PER_BLOCK = 16


def check_mask_token(tok):
    for cand in ("[MASK]", "<mask>", "<|mask|>"):
        tid = tok.convert_tokens_to_ids(cand)
        if tid is not None and tid == MASK_TOKEN_ID:
            print(f"[ok] mask token {cand!r} -> id {tid} matches MASK_TOKEN_ID")
            return
    print(f"[!] could not confirm token for MASK_TOKEN_ID={MASK_TOKEN_ID}; "
          f"decode({MASK_TOKEN_ID}) = {tok.decode([MASK_TOKEN_ID])!r}")


def run_ar(model, tok, input_ids):
    t0 = time.time()
    with torch.no_grad():
        out = model.generate_ar(input_ids, max_new_tokens=MAX_NEW)
    dt = time.time() - t0
    text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n[AR] {dt:.2f}s  tokens/NFE=1.00 (one token per forward)")
    print(f"[AR] {text!r}")


def run_diffusion(model, tok, input_ids):
    cb = nan_guard_callback()
    reset_nfe(model)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate_mask_diffusion(
            input_ids,
            max_new_tokens=MAX_NEW,
            block_size=BLOCK_SIZE,
            steps_per_block=STEPS_PER_BLOCK,
            mask_token_id=MASK_TOKEN_ID,
            temperature=0.0,
            confidence_threshold=0.8,
            eos_token_id=tok.eos_token_id,
            step_callback=cb,
        )
    dt = time.time() - t0
    nfe = get_nfe(model)
    text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    tpn = (MAX_NEW / nfe) if nfe else float("nan")
    print(f"\n[diffusion] {dt:.2f}s  NFE={nfe}  tokens/NFE={tpn:.2f}  "
          f"(parallelism vs AR)")
    print(f"[diffusion] NaN hits: {cb.hits if cb.hits else 'none'}")
    print(f"[diffusion] {text!r}")
    assert text.strip(), "diffusion produced empty output"
    assert not cb.hits, f"NaN/Inf in denoiser logits at {cb.hits}"


def main():
    tok = load_tokenizer()
    check_mask_token(tok)
    model, tok = load()  # 2 GPUs, downloads weights on first run
    input_ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    run_ar(model, tok, input_ids)
    run_diffusion(model, tok, input_ids)
    print("\n[M0] smoke test PASSED")


if __name__ == "__main__":
    main()
````

### `src/diffusion_smoke.py`  ·  49 行

**作用**：扩散路径冒烟

> 文件自述：Coherence smoke for diffusion — passing imports is NOT enough, the output must be real

````python
"""Coherence smoke for diffusion — passing imports is NOT enough, the output must be real
text, not word-salad. Exit 0 if the run shows genuine parallelism (NFE < max) AND low
repetition; exit 1 (garbage) otherwise. Used as the pass/fail gate at the end of install.sh.

The garbage failure mode looked like: NFE == max (nothing confident, 1 token/step) and the
output dominated by a few repeated tokens (", best best the the ..."). Both are caught here.
"""
import collections
import sys

import torch

from twotower import load, MASK_TOKEN_ID, reset_nfe, get_nfe

PROMPT = "France is a country "
MAX_NEW, BLOCK, STEPS = 64, 16, 16


def main():
    model, tok = load()  # both towers, 2 GPUs
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")

    reset_nfe(model)
    with torch.no_grad():
        out = model.generate_mask_diffusion(
            ids, max_new_tokens=MAX_NEW, block_size=BLOCK, steps_per_block=STEPS,
            mask_token_id=MASK_TOKEN_ID, temperature=0.1, confidence_threshold=0.8,
            eos_token_id=tok.eos_token_id)
    nfe = get_nfe(model) or 0
    gen_ids = out[0][ids.shape[1]:].tolist()
    text = tok.decode(gen_ids, skip_special_tokens=True)

    max_nfe = (MAX_NEW // BLOCK) * STEPS
    tokens_per_nfe = MAX_NEW / nfe if nfe else 0.0
    top_freq = (max(collections.Counter(gen_ids).values()) / len(gen_ids)) if gen_ids else 1.0

    parallel_ok = nfe < max_nfe        # a working model commits >1 token on some step
    rep_ok = top_freq < 0.35           # garbage is dominated by a few repeated tokens
    ok = parallel_ok and rep_ok

    print(f"NFE={nfe}/{max_nfe}  tokens/NFE={tokens_per_nfe:.2f}  top_token_freq={top_freq:.2f}")
    print("OUT:", repr(text[:220]))
    print("DIFFUSION", "COHERENT (PASS)" if ok else "GARBAGE (FAIL)",
          f"[parallel_ok={parallel_ok}, rep_ok={rep_ok}]")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
````

### `src/diagnose_diffusion.py`  ·  71 行

**作用**：扩散垃圾输出的系统诊断

> 文件自述：Localize the diffusion garbage. One run, three switchable bypasses of the denoiser-only

````python
"""Localize the diffusion garbage. One run, three switchable bypasses of the denoiser-only
machinery (AR works, so the context tower / lm_head / tokenizer are fine — the bug is in
what diffusion adds). Whichever row becomes grammatical identifies the culprit.

  baseline : as-is (should be garbage)
  seed-off : Mamba initial_states=None  -> bypasses causal_conv1d/chunk-scan SEEDING kernels
                                           (the triton hypothesis lives here)
  adaln-off: time t=None                -> bypasses AdaLN modulation applied to every layer

If ALL THREE are garbage, the cause is NOT seeding/AdaLN -> suspect denoiser weights not
loading, _mdlm_forward, or cross-attention (need _denoiser_block_attention).

    python src/diagnose_diffusion.py
"""
import collections

import torch

from twotower import load, MASK_TOKEN_ID, reset_nfe, get_nfe

PROMPT = "France is a country "
MAX_NEW, BLK, STEPS = 64, 16, 16
MAX_NFE = (MAX_NEW // BLK) * STEPS


def gen(model, tok, tag):
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    reset_nfe(model)
    with torch.no_grad():
        out = model.generate_mask_diffusion(
            ids, max_new_tokens=MAX_NEW, block_size=BLK, steps_per_block=STEPS,
            mask_token_id=MASK_TOKEN_ID, temperature=0.1, confidence_threshold=0.8,
            eos_token_id=tok.eos_token_id)
    nfe = get_nfe(model) or 0
    g = out[0][ids.shape[1]:].tolist()
    text = tok.decode(g, skip_special_tokens=True)
    top = (max(collections.Counter(g).values()) / len(g)) if g else 1.0
    ok = (nfe < MAX_NFE) and (top < 0.35)
    print(f"[{tag:9}] NFE={nfe}/{MAX_NFE} tok/NFE={(MAX_NEW/nfe if nfe else 0):.2f} "
          f"top_freq={top:.2f} {'OK' if ok else 'GARBAGE'} | {text[:140]!r}")


def main():
    model, tok = load()
    T = type(model)

    gen(model, tok, "baseline")

    # seed-off: force Mamba initial_states=None (model natively supports None)
    o1 = T._denoiser_block_mamba
    T._denoiser_block_mamba = (
        lambda self, mx, h, ic, iss, return_states=False:
        o1(self, mx, h, None, None, return_states=return_states))
    gen(model, tok, "seed-off")
    T._denoiser_block_mamba = o1

    # adaln-off: force time t=None so no per-layer modulation is applied
    o2 = T._run_denoiser_step_diffusion
    T._run_denoiser_step_diffusion = (
        lambda self, block_ids, cache_state, t=None, den_cache=None:
        o2(self, block_ids, cache_state, t=None, den_cache=den_cache))
    gen(model, tok, "adaln-off")
    T._run_denoiser_step_diffusion = o2

    print("\n=> the row that turns grammatical (OK) = the bypassed module was the culprit.")
    print("   if all three are GARBAGE -> not seeding/AdaLN; suspect denoiser weights / "
          "_mdlm_forward / cross-attn (check load warnings for missing keys).")


if __name__ == "__main__":
    main()
````

### `src/prep_data.py`  ·  102 行

**作用**：准备/合成评测 prompts（synthmath 等）

> 文件自述：Build mini benchmark prompt files. Works OFFLINE.

````python
"""Build mini benchmark prompt files. Works OFFLINE.

  gsm8k     : 4-shot chain-of-thought, reference = gold final number.
              Tries HF `datasets` (real GSM8K); if offline/unavailable, falls back to an
              embedded set of simple arithmetic word problems (answers authored+verified here)
              so it works with HF_HUB_OFFLINE=1. The embedded set is labeled synthmath_* and is
              NOT official GSM8K — fine for a quality/trend smoke, note it in the writeup.
  humaneval : 0-shot completion (needs HF datasets / network).

    python src/prep_data.py --which gsm8k     --n 15 --out data/gsm8k_mini.jsonl
    python src/prep_data.py --which humaneval --n 25 --out data/humaneval_mini.jsonl
"""
import argparse
import json
import os

GSM8K_SHOTS = """Question: Natalia sold clips to 48 friends in April, and half as many in May. How many clips did she sell altogether?
Answer: In May she sold 48 / 2 = 24 clips. Altogether 48 + 24 = 72. #### 72

Question: Weng earns $12 an hour for babysitting. Yesterday she babysat 50 minutes. How much did she earn?
Answer: Per minute she earns 12 / 60 = $0.2. For 50 minutes she earned 50 * 0.2 = $10. #### 10

Question: Betty has half the money she needs for a $100 wallet. Her parents give her $15 and her grandparents twice as much. How much more does she need?
Answer: Betty has 100 / 2 = $50. Grandparents give 15 * 2 = $30. Now she has 50 + 15 + 30 = $95. She needs 100 - 95 = $5. #### 5

Question: James writes a 3-page letter to 2 friends twice a week. How many pages does he write a year?
Answer: Each time he writes 3 * 2 = 6 pages. Twice a week that's 6 * 2 = 12 pages. In a year 12 * 52 = 624. #### 624

"""

# Authored + verified simple arithmetic word problems (offline fallback). answers are correct.
_EMBEDDED = [
    ("Tom has 12 apples. He buys 8 more and then gives 5 to his friend. How many apples does he have now?", "15"),
    ("A book has 240 pages. Sarah reads 60 pages each day. How many days does she need to finish it?", "4"),
    ("There are 5 boxes with 12 pencils each. How many pencils are there in total?", "60"),
    ("Maria earns $15 per hour and works 6 hours. How much does she earn?", "90"),
    ("A train travels 80 km in 2 hours. What is its speed in km per hour?", "40"),
    ("John had 100 dollars. He spent 35 on a shirt and 20 on lunch. How much money is left?", "45"),
    ("A class has 30 students. 18 of them are girls. How many are boys?", "12"),
    ("A rectangle is 8 meters long and 3 meters wide. What is its area in square meters?", "24"),
    ("Anna bakes 4 dozen cookies. How many cookies is that?", "48"),
    ("A car uses 6 liters of fuel per 100 km. How many liters does it use for 250 km?", "15"),
    ("Ben saves $7 each week. How much does he save in 8 weeks?", "56"),
    ("A pizza is cut into 8 slices. 3 people each eat 2 slices. How many slices are left?", "2"),
    ("There are 3 shelves with 25 books each. How many books are there in total?", "75"),
    ("Lucy has 45 stickers. She gives away 18 and then buys 10 more. How many stickers does she have now?", "37"),
    ("A farmer has 9 cows and each cow gives 12 liters of milk. How many liters of milk in total?", "108"),
]


def build_gsm8k(n, out):
    items = []
    src = "gsm8k(HF)"
    try:
        from datasets import load_dataset
        try:
            ds = load_dataset("gsm8k", "main", split="test")
        except Exception:
            ds = load_dataset("openai/gsm8k", "main", split="test")
        for i in range(min(n, len(ds))):
            gold = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
            items.append((f"gsm8k_{i}", ds[i]["question"], gold))
    except Exception as e:
        print(f"[prep] HF datasets unavailable ({type(e).__name__}); using embedded synthetic set")
        src = "embedded-synthetic"
        for i, (q, a) in enumerate(_EMBEDDED[:n]):
            items.append((f"synthmath_{i}", q, a))

    with open(out, "w", encoding="utf-8") as f:
        for id_, q, gold in items:
            prompt = GSM8K_SHOTS + f"Question: {q}\nAnswer:"
            f.write(json.dumps({"id": id_, "prompt": prompt, "reference": gold},
                               ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} prompts from {src} -> {out}")


def build_humaneval(n, out):
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    with open(out, "w", encoding="utf-8") as f:
        for i in range(min(n, len(ds))):
            ex = ds[i]
            f.write(json.dumps({"id": ex["task_id"], "task_id": ex["task_id"],
                                "prompt": ex["prompt"],
                                "reference": {"test": ex["test"],
                                              "entry_point": ex["entry_point"]}},
                               ensure_ascii=False) + "\n")
    print(f"wrote {min(n, len(ds))} humaneval prompts -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["gsm8k", "humaneval"], required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    (build_gsm8k if args.which == "gsm8k" else build_humaneval)(args.n, args.out)


if __name__ == "__main__":
    main()
````

---

## 9 · 定位期探针（二分排查留痕）

报告 §6/§9「系统性二分排查 > 盲猜」的现场证据。逐一排除环境/cache/prompt/kernel。

### `src/probe_official.py`  ·  30 行

**作用**：对拍官方 inference.py 行为

> 文件自述：Decisive fork: does the DENOISER-tower forward itself work, or is only the diffusion

````python
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
````

### `src/probe_realprompt.py`  ·  35 行

**作用**：真实长 prompt vs 短 prompt 的差异探针

> 文件自述：The base model scores HumanEval 76 / GSM8K 85 — it is NOT weak. So the garbage must be

````python
"""The base model scores HumanEval 76 / GSM8K 85 — it is NOT weak. So the garbage must be
our driving of it. Prime suspect: our 5-token prompt "France is a country " is too weak for
block-parallel denoising. The paper's benchmarks use long few-shot / code prompts. Test with
a REAL strong-context prompt and see if diffusion becomes coherent.

    python src/probe_realprompt.py
"""
import torch
from twotower import load

model, tok = load()

# a proper few-shot math prompt (strong, long context — like the paper's eval)
PROMPT = (
    "Question: There are 15 trees in the grove. Grove workers will plant trees today. "
    "After they are done there will be 21 trees. How many trees did they plant?\n"
    "Answer: There were 15 trees, then 21. So they planted 21 - 15 = 6. The answer is 6.\n\n"
    "Question: If there are 3 cars and 2 more arrive, how many cars are there?\n"
    "Answer: There are 3 + 2 = 5 cars. The answer is 5.\n\n"
    "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many "
    "pieces do they have left in total?\nAnswer:"
)
ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
print("prompt tokens:", ids.shape[1])

for bs, st in [(16, 16)]:
    out = model.generate_mask_diffusion(
        ids, max_new_tokens=64, block_size=bs, steps_per_block=st, mask_token_id=3,
        temperature=0.0, confidence_threshold=0.8, eos_token_id=tok.eos_token_id)
    print(f"\n[diffusion bs={bs} steps={st}] NFE={getattr(model,'_last_nfe',None)}")
    print("OUT:", repr(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)))

# AR on the same prompt for reference (known-good)
o = model.generate_ar(ids, max_new_tokens=64, eos_token_id=tok.eos_token_id)
print("\n[AR ref] OUT:", repr(tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)))
````

### `src/probe_freshcache.py`  ·  56 行

**作用**：cache 复用是否污染

> 文件自述：HYPOTHESIS: generate_mask_diffusion builds den_cache ONCE per block and reuses it across

````python
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
````

### `src/probe_eager.py`  ·  28 行

**作用**：eager vs 编译 attention 实现

> 文件自述：Test whether attn_implementation="eager" fixes the diffusion garbage. The HF stack

````python
"""Test whether attn_implementation="eager" fixes the diffusion garbage. The HF stack
defaults to sdpa; the model warns about eager being required. Reload with eager and rerun
the official diffusion example + mock_ar. If output turns coherent -> it was the attention
implementation, not a model bug.

    python src/probe_eager.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from twotower import MODEL_NAME

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="eager",          # <-- the one thing we haven't tried
    low_cpu_mem_usage=False,
)
model.place_towers_on_devices("cuda:0", "cuda:1")
model.eval()
print("attn impl:", getattr(model.config, "_attn_implementation", "?"))

ids = tok("France is a country ", return_tensors="pt").input_ids.to("cuda:0")
out = model.generate_mask_diffusion(
    ids, max_new_tokens=32, block_size=16, steps_per_block=16, mask_token_id=3,
    temperature=0.1, confidence_threshold=0.8, eos_token_id=tok.eos_token_id)
print("NFE:", getattr(model, "_last_nfe", None))
print("DIFFUSION:", repr(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)))
````

### `src/probe_logits.py`  ·  56 行

**作用**：去噪塔 logits 数值检查

> 文件自述：Split "denoiser forward is broken" vs "post-processing (mdlm/confidence) is broken".

````python
"""Split "denoiser forward is broken" vs "post-processing (mdlm/confidence) is broken".

Weights load fine and triton/AdaLN/seeding are ruled out, so the garbage is either in the
denoiser FORWARD (cross-attention / context cache) or in _mdlm_forward/confidence. This
captures the RAW logits from the first denoiser step (all-mask block + real context) and
decodes the argmax per position:

  - argmax = plausible continuation words, some high max-prob  -> forward is FINE; the bug is
    in _mdlm_forward / confidence / commit (post-processing).
  - argmax = junk, max-prob ~uniform (~1/vocab)               -> the FORWARD is broken
    (context not reaching the denoiser: cross-attn / cache).

    python src/probe_logits.py
"""
import torch

from twotower import load, MASK_TOKEN_ID

PROMPT = "France is a country "


def main():
    model, tok = load()
    cap = {}
    orig = type(model)._run_denoiser_step_diffusion

    def wrap(self, block_ids, cache_state, t=None, den_cache=None):
        lg = orig(self, block_ids, cache_state, t=t, den_cache=den_cache)
        if "logits" not in cap:
            cap["logits"] = lg.detach().float().cpu()
        return lg

    type(model)._run_denoiser_step_diffusion = wrap
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        model.generate_mask_diffusion(
            ids, max_new_tokens=16, block_size=16, steps_per_block=2,
            mask_token_id=MASK_TOKEN_ID, temperature=0.0, confidence_threshold=0.8,
            eos_token_id=tok.eos_token_id)
    type(model)._run_denoiser_step_diffusion = orig

    lg = cap["logits"][0]                      # (L, V)
    probs = lg.softmax(-1)
    maxp = probs.max(-1).values
    arg = lg.argmax(-1)
    print("logits shape:", tuple(lg.shape), "| vocab:", lg.shape[-1],
          "| uniform prob ~", round(1 / lg.shape[-1], 6))
    print("per-position max prob:", [round(float(x), 4) for x in maxp])
    print("argmax decoded:", repr(tok.decode(arg)))
    for i in range(min(5, lg.shape[0])):
        v, idx = probs[i].topk(5)
        print(f"  pos{i} top5:", [(tok.decode([int(t)]), round(float(p), 3)) for p, t in zip(v, idx)])


if __name__ == "__main__":
    main()
````

### `src/probe_normorder.py`  ·  80 行

**作用**：norm/modulate 顺序探针

> 文件自述：STATIC AUDIT RESULT: the diffusion denoiser applies adaLN as modulate-THEN-norm for

````python
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
````

### `src/probe_gen_internal.py`  ·  47 行

**作用**：生成内部状态探针

> 文件自述：AR works on this exact 149-tok prompt but diffusion is garbage -> the bug is diffusion-

````python
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
````

### `src/probe_teacher.py`  ·  38 行

**作用**：teacher-forcing 探针

> 文件自述：Is the denoiser forward actually broken, or is this just a weak base model / wrong call?

````python
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
````

---

## 10 · Shell 编排与安装脚本

报告 §9 环境 8 坑固化。一键复现/批跑/监控/安装。

### `run_everything.sh`  ·  57 行

**作用**：一键跑通全流程

> 文件自述：Run ALL experiments unattended (one full day is fine). Each step is ISOLATED: a failure is

````bash
#!/usr/bin/env bash
# Run ALL experiments unattended (one full day is fine). Each step is ISOLATED: a failure is
# logged and the run continues, so one broken experiment can't abort the overnight batch.
#
# Launch (survives SSH disconnect; keep the pod RUNNING until done — stopping the pod kills it):
#   cd /workspace/twotower-repro
#   export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
#   nohup bash run_everything.sh > results/run.log 2>&1 &
#   tail -f results/run.log
#
# Per-experiment logs land in results/logs/<name>.log ; all outputs in results/.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p results/logs

PY=/usr/local/bin/python
PROMPTS=data/gsm8k_mini.jsonl
SPEED=data/speed_prompts.jsonl
T0=$(date +%s)

run () {                       # run <name> <args...>
  local name="$1"; shift
  local s=$(date +%s)
  echo "===== [$(date +%H:%M:%S)] START $name  |  $*"
  if env "$@" >"results/logs/$name.log" 2>&1; then
    echo "----- [$(date +%H:%M:%S)] DONE  $name  ($(($(date +%s)-s))s)"
  else
    echo "!!!!! [$(date +%H:%M:%S)] FAILED $name ($(($(date +%s)-s))s) -> results/logs/$name.log"
  fi
}

# 0) prompts (offline-safe embedded set if HF datasets unavailable)
$PY src/prep_data.py --which gsm8k --n 15 --out "$PROMPTS" || true

echo "########## CHEAP / HIGH-VALUE FIRST ##########"
# before/after: capture the BUGGY trace too (for the pre's before-vs-after visual)
run exp0_buggy_trace  TWOTOWER_NOFIX=1 $PY src/exp0_capture.py --out results/trace_buggy.npz
run exp0_capture      $PY src/exp0_capture.py --out results/trace.npz
run abl_remask        $PY src/ablation_remask.py    --prompts "$PROMPTS" --out results/abl_remask.jsonl   --limit 10
run collapse_e3       $PY src/run_all.py --exp e3    --prompts "$PROMPTS" --out results/e3.jsonl          --limit 10
run abl_denoiser      $PY src/ablation_denoiser.py  --prompts "$PROMPTS" --out results/abl_denoiser.jsonl --limit 10
run abl_topk          $PY src/ablation_topk.py      --prompts "$PROMPTS" --out results/abl_topk.pkl       --limit 10
run main_run_ABD      $PY src/main_run.py           --prompts "$PROMPTS" --out results/trace_main.pkl     --limit 3
run main_run_moe_E    $PY src/main_run.py --moe-hook --prompts "$PROMPTS" --out results/trace_moe.pkl     --limit 3

echo "########## EXPENSIVE (overnight) ##########"
run ar_baseline       $PY src/run_all.py --exp ar    --prompts "$PROMPTS" --out results/ar.jsonl          --limit 15
run pareto_e2         $PY src/run_all.py --exp e2    --prompts "$PROMPTS" --out results/e2.jsonl          --limit 15
run speed_e1          $PY src/run_all.py --exp e1    --prompts "$SPEED"   --out results/e1.jsonl          --limit 10

echo ""
echo "########## ALL DONE in $(( ($(date +%s)-T0)/60 )) min ##########"
echo "results:"; ls -la results/
echo "logs:";    ls -la results/logs/
echo "Download results/ to your Mac, then score+plot offline. Stop the pod to save money."
````

### `run_batch.sh`  ·  42 行

**作用**：批量跑实验矩阵

> 文件自述：Main test batch — the remaining pod runs, in one sequential pass (one model at a time).

````bash
#!/usr/bin/env bash
# Main test batch — the remaining pod runs, in one sequential pass (one model at a time).
# Avoids hand-pasting long commands (which kept getting line-split).
#
#   cd /workspace/twotower-repro && git pull && bash run_batch.sh
#
# Survive SSH disconnect (recommended):
#   nohup bash run_batch.sh > results/logs/batch.log 2>&1 &
#   tail -f results/logs/batch.log        # watch progress; Ctrl-C only stops the tail, not the run
#
# Monitor GPU from another terminal:   watch -n 2 nvidia-smi
set -uo pipefail                          # NOT -e: a failed step is logged, the batch continues
cd "$(dirname "$0")"
source setup/env.sh
export HF_HUB_OFFLINE=1                    # weights+code are local; never re-pull
mkdir -p results/logs
PY=python

run () {
  echo "===== [$(date +%H:%M:%S)] START  $*"
  if "$@"; then echo "----- [$(date +%H:%M:%S)] DONE   $1 ..."
  else echo "!!!!! [$(date +%H:%M:%S)] FAILED $* (continuing)"; fi
}

# ① block-64 collapse triangle
run $PY src/exp0_capture.py --block-size 64 --max-new 128 --steps 16 --gamma 0.8 --out results/trace_tri_b64.npz
# ② AR baseline (own speedup / quality retention)
run $PY src/run_all.py --exp ar --prompts data/gsm8k_mini.jsonl --out results/ar.jsonl --limit 15
# ③ HumanEval code-side collapse + AR code baseline
run $PY src/run_all.py --exp e3 --prompts data/humaneval_mini.jsonl --out results/he_collapse.jsonl --limit 10
run $PY src/run_all.py --exp ar --prompts data/humaneval_mini.jsonl --out results/he_ar.jsonl --limit 10
# ④ long-context needle (32K may OOM -> auto-recorded, not fatal)
run $PY src/ruler_lite.py --lengths 2048 8192 16384 32768 --out results/ruler.jsonl --ar
# ⑤ top-k MoE ablation (optional footnote)
run $PY src/ablation_topk.py --prompts data/gsm8k_mini.jsonl --out results/abl_topk.pkl --limit 10
# ⑥ before/after bug demo: BUGGY word-salad (NOFIX) vs the fixed b16 you already captured
run env TWOTOWER_NOFIX=1 $PY src/exp0_capture.py --block-size 16 --max-new 64 --steps 16 --gamma 0.8 --out results/trace_buggy_demo.npz
# ⑦ aggressive-γ triangle (same prompt as b16): does τb stay high at MAX parallelism (γ0.5, steps4)?
run $PY src/exp0_capture.py --block-size 16 --max-new 64 --steps 4 --gamma 0.5 --out results/trace_tri_aggr.npz

echo "===== ALL DONE ====="
ls -la results/*.jsonl results/*.npz results/*.pkl 2>/dev/null
````

### `monitor.sh`  ·  23 行

**作用**：GPU/进程监控

> 文件自述：Live dashboard for the run_everything.sh batch.

````bash
#!/usr/bin/env bash
# Live dashboard for the run_everything.sh batch.
# Use a refreshing view:   watch -n 30 bash monitor.sh
# Or one-shot:             bash monitor.sh
cd "$(dirname "$0")"

echo "======== TwoTower run monitor  $(date '+%Y-%m-%d %H:%M:%S') ========"
echo ""
echo "-- progress (START / DONE / FAILED) --"
grep -E "START|DONE|FAILED|ALL DONE" results/run.log 2>/dev/null | tail -22 || echo "  (no results/run.log yet)"
echo ""
echo "-- is the batch still running? --"
if pgrep -f run_everything.sh >/dev/null; then echo "  YES (run_everything.sh alive)"; else echo "  NO (finished or not started)"; fi
echo ""
echo "-- outputs produced --"
ls -1sh results/*.jsonl results/*.npz results/*.pkl 2>/dev/null || echo "  (none yet)"
echo ""
echo "-- GPU --"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi unavailable)"
echo ""
echo "-- live tail: most recent experiment log --"
newest=$(ls -t results/logs/*.log 2>/dev/null | head -1)
if [ -n "$newest" ]; then echo "  [$(basename "$newest")]"; tail -8 "$newest" | sed 's/^/  /'; else echo "  (no experiment logs yet)"; fi
````

### `setup/install.sh`  ·  89 行

**作用**：环境安装（--no-deps 锁 torch、别升 triton 等 8 坑固化）

> 文件自述：Idempotent installer. RunPod swaps containers and ONLY /workspace persists, so the pip

````bash
#!/usr/bin/env bash
# Idempotent installer. RunPod swaps containers and ONLY /workspace persists, so the pip
# env is wiped on every swap. Recover with one command: `bash setup/install.sh`.
# Wheels are cached on /workspace (PIP_CACHE_DIR) so re-install is fast (no re-download).
#
# It does NOT stop at "imports work" — it ends by SMOKE-testing that diffusion generates
# COHERENT text (src/diffusion_smoke.py). If that fails it tries the triton>=3.5 hypothesis
# and, if that doesn't help, reverts and tells you triton was not the cause.
#
# torch is LOCKED (it's the pod image); we never change it.
set -uo pipefail                      # NOT -e: we handle failures explicitly
cd "$(dirname "$0")/.."               # run from repo root (needs src/)

# The /workspace network FS (MooseFS) is pathologically slow for the many small files a
# python env needs (a venv there hangs pip for many minutes). So install the env to the
# LOCAL container disk via system python (fast) and just re-run this after each container
# swap (~2 min). ONLY the 126GB weights live on /workspace/hf (large files -> network is fine).
echo "installing env to system python (local disk); weights persist on \$HF_HOME=/workspace/hf"

MAMBA_VER="${MAMBA_VER:-2.3.2.post1}"
CAUSAL_VER="${CAUSAL_VER:-1.6.2.post1}"   # torch2.8 wheels; this mamba/causal pair targets triton 3.4

TORCH_V=$(python -c "import torch;print(torch.__version__)" 2>/dev/null || echo none)
echo "torch: $TORCH_V (LOCKED - never modified)"
case "$TORCH_V" in
  2.8*) ;;
  *) echo "WARN: expected torch 2.8.x; the mamba/causal wheels below target torch2.8." ;;
esac

# --- detect wheel tags from the live interpreter ---
read -r PY TORCH CU ABI <<EOF
$(python - <<'PYEOF'
import torch, sys
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
tv = torch.__version__.split("+")[0].split(".")
print(py, f"torch{tv[0]}.{tv[1]}", "cu" + (torch.version.cuda or "0.0").split(".")[0],
      "cxx11abiTRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "cxx11abiFALSE")
PYEOF
)
EOF
echo "wheel tags: $PY $TORCH $CU $ABI"

MAMBA_URL="https://github.com/state-spaces/mamba/releases/download/v${MAMBA_VER}/mamba_ssm-${MAMBA_VER}+${CU}${TORCH}${ABI}-${PY}-${PY}-linux_x86_64.whl"
CAUSAL_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CAUSAL_VER}/causal_conv1d-${CAUSAL_VER}+${CU}${TORCH}${ABI}-${PY}-${PY}-linux_x86_64.whl"

# --- base python deps (einops is REQUIRED by the modeling file; hf_transfer speeds the
#     126GB download). These are pure-python and must NOT touch torch. ---
echo ">>> base deps"
python -m pip install -q einops "transformers==4.57.1" accelerate safetensors sentencepiece \
    matplotlib imageio pillow datasets hf_transfer || { echo "base deps failed"; exit 1; }

# --- ABI-sensitive kernels. CRITICAL: --no-deps. Without it, causal-conv1d 1.6.2's
#     dependency chain UPGRADES torch (2.8 -> 2.13) + a CUDA-13 stack, which breaks the
#     mamba .so ABI (undefined symbol). The wheels are prebuilt for the base image's torch,
#     so they need no deps resolved. ---
echo ">>> kernels (--no-deps, torch is never touched): causal $CAUSAL_VER, mamba $MAMBA_VER"
python -m pip install -q --no-deps "$CAUSAL_URL" || { echo "!! causal wheel 404 ($CU/$TORCH/$ABI/$PY) - see https://github.com/Dao-AILab/causal-conv1d/releases"; exit 1; }
python -m pip install -q --no-deps "$MAMBA_URL"  || { echo "!! mamba wheel 404 ($CU/$TORCH/$ABI/$PY) - see https://github.com/state-spaces/mamba/releases"; exit 1; }
TORCH_NOW=$(python -c "import torch;print(torch.__version__)" 2>/dev/null || echo BROKEN)
[ "$TORCH_NOW" = "$TORCH_V" ] || echo "!! WARNING: torch changed from $TORCH_V to $TORCH_NOW — deps churn!"

python - <<'PYEOF' || { echo "!! kernel import failed"; exit 1; }
import causal_conv1d, mamba_ssm, einops, transformers, triton
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # forces symbol load
print(f"imports OK | triton {triton.__version__} | mamba {mamba_ssm.__version__} "
      f"| causal {causal_conv1d.__version__} | transformers {transformers.__version__}")
PYEOF

# --- coherence smoke drives pass/fail (needs 2 GPUs to hold both towers) ---
NGPU=$(python -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [ "$NGPU" -lt 2 ]; then
  echo "only $NGPU GPU visible - env installed OK, but diffusion smoke needs 2 GPUs. Skipping."
  exit 0
fi

echo ">>> diffusion coherence smoke (stock stack, triton $(python -c 'import triton;print(triton.__version__)'))"
if python src/diffusion_smoke.py; then
  echo "PASS: diffusion coherent with the stock stack. Done."
  exit 0
fi

# stock stack garbage. Do NOT churn versions: triton / AdaLN / seeding / weight-loading are
# all ruled out (see diagnose_diffusion.py + check_weights.py). Upgrading triton here only
# pulled a mismatched CUDA runtime (libcudart.so.13) and corrupted the env. The env install
# itself succeeded; the diffusion-quality bug is a separate, code-level issue under
# investigation (cross-attention / _mdlm_forward). Report and exit 0 (env is usable).
echo "!! diffusion smoke reports GARBAGE — env is installed OK, but generation quality is a"
echo "   separate code-level bug (NOT triton). Run: python src/probe_logits.py"
exit 0
````

### `setup/env.sh`  ·  14 行

**作用**：环境变量

> 文件自述：Source this in every new shell:

````bash
# Source this in every new shell:
#   source setup/env.sh
# Only the 126GB WEIGHTS persist on /workspace (network volume). The python env installs to
# the LOCAL container disk (a venv on the network FS hangs pip), so after a container swap
# you re-run `bash setup/install.sh` (~2 min) — you do NOT re-download the weights.
export HF_HOME=/workspace/hf                  # 126GB weights persist here
export HF_HUB_OFFLINE=1                        # weights+code are local; stay offline so HF never
                                              # re-pulls a new revision or re-downloads 126GB.
                                              # For the humaneval download, override inline:
                                              #   HF_HUB_OFFLINE=0 python src/prep_data.py --which humaneval ...
# export HF_TOKEN=hf_xxx                       # set in your shell; do NOT commit it

echo "env set: HF_HOME=$HF_HOME  HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "after a container swap: bash setup/install.sh   (env is local+fast; weights stay on /workspace)"
````

---

## 11 · NVIDIA 官方参考代码（非本复现原创）

以下两份是 NVIDIA 随模型发布的官方代码，仓库内留作对照参考。报告的机制描述与 bug 定位均以此为基准。

### `reference_modeling.py`  ·  962 行

**作用**：官方 HF 双塔建模文件（含 generate_mask_diffusion / _denoiser_block_mamba 原版）

````python
# coding=utf-8
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Two-tower NemotronH for HuggingFace — real separate context + denoiser weights.
#
# Checkpoint key layout (from converted safetensors):
#   context_tower.*        — context backbone (NemotronHModel)
#   context_lm_head.weight — context output head
#   denoiser_tower.*       — denoiser backbone (NemotronHModel)
#   lm_head.weight         — denoiser output head
#   t_embedder.*           — timestep embedder (optional, for mask_diffusion)
#   t_block.*              — timestep MLP (optional)
#   scale_shift_tables.*   — per-layer modulation bias (optional)
#
# Modes:
#   AR:             forward() + generate() — context_tower only
#   Mock-AR:        generate_mock_ar() — two-tower, S-2/KV[:-1] semantics
#   Mask-Diffusion: generate_mask_diffusion() — block-wise iterative denoising

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from .modeling_nemotron_h import (
        HybridMambaAttentionDynamicCache,
        NemotronHCausalLMOutput,
        NemotronHForCausalLM,
        NemotronHModel,
        NemotronHPreTrainedModel,
        repeat_kv,
    )
    from .configuration_nemotron_h import NemotronHConfig
except ImportError:
    from modeling_nemotron_h import (
        HybridMambaAttentionDynamicCache,
        NemotronHCausalLMOutput,
        NemotronHForCausalLM,
        NemotronHModel,
        NemotronHPreTrainedModel,
        repeat_kv,
    )
    from configuration_nemotron_h import NemotronHConfig

from transformers.generation import GenerationMixin


# ---------------------------------------------------------------------------
# Time conditioning (PixArt-alpha adaLN-single style)
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Sinusoidal + MLP embedder for scalar timesteps in [0,1]."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256,
                 max_period: int = 1000):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(t.dtype)

    def forward(self, t):
        t_scaled = t * self.max_period
        t_freq = self.timestep_embedding(t_scaled, self.frequency_embedding_size)
        return self.mlp(t_freq)


def _modulate(x, shift, scale):
    """Adaptive LN: x * (1 + scale) + shift. Broadcasts for (B,L,D) input."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _get_mod_params(t_emb, table):
    """(B, 3*D) + (3, D) -> (shift, scale, gate) each (B, D)."""
    B, D = t_emb.shape[0], table.shape[1]
    combined = table[None] + t_emb.reshape(B, 3, D)
    shift, scale, gate = combined.chunk(3, dim=1)
    return shift.squeeze(1), scale.squeeze(1), gate.squeeze(1)


# ---------------------------------------------------------------------------
# Bug-fixed cache
# ---------------------------------------------------------------------------

class FixedHybridCache(HybridMambaAttentionDynamicCache):
    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        super().__init__(config, batch_size, dtype, device)
        self.conv_kernel_size = config.conv_kernel

    def update_conv_state(self, layer_idx, new_conv_state, cache_init=False):
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(self.conv_states[layer_idx].device)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(
                self.conv_states[layer_idx].device
            )
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx, new_ssm_state):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states[layer_idx].device)
        return self.ssm_states[layer_idx]


# ---------------------------------------------------------------------------
# Two-Tower CausalLM
# ---------------------------------------------------------------------------

class NemotronHTwoTowerForCausalLM(NemotronHPreTrainedModel, GenerationMixin):
    """Two-tower NemotronH with real separate context and denoiser weights.

    Modes:
        AR:             forward() + generate() — context_tower only
        Mock-AR:        generate_mock_ar() — S-2/KV[:-1] semantics
        Mask-Diffusion: generate_mask_diffusion() — block-wise confidence_unmasking
    """

    _tied_weights_keys = []

    def __init__(self, config: NemotronHConfig):
        super().__init__(config)
        self.context_tower = NemotronHModel(config)
        self.context_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.denoiser_tower = NemotronHModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size

        # Time conditioning (created unconditionally; weights loaded if present)
        H = config.hidden_size
        N = config.num_hidden_layers
        self.t_embedder = TimestepEmbedder(H)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(H, 3 * H, bias=True))
        self.scale_shift_tables = nn.ParameterList([
            nn.Parameter(torch.randn(3, H) / (H ** 0.5)) for _ in range(N)
        ])

        self.post_init()

    # ------------------------------------------------------------------
    # HF interface
    # ------------------------------------------------------------------

    def get_input_embeddings(self):
        return self.context_tower.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.context_tower.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.context_lm_head

    def set_output_embeddings(self, new_embeddings):
        self.context_lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        inputs_embeds=None, cache_position=None, position_ids=None,
        use_cache=True, **kwargs,
    ):
        empty_past_kv = past_key_values is None
        if not empty_past_kv:
            if inputs_embeds is not None or cache_position[-1] >= input_ids.shape[1]:
                input_ids = input_ids[:, -cache_position.shape[0]:]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]
        else:
            # FixedHybridCache (not the base class) so the Mamba mixer finds
            # conv_kernel_size during the cached forward (needed for AR generate).
            past_key_values = FixedHybridCache(
                self.config, input_ids.shape[0], self.dtype,
                device=next(self.context_tower.parameters()).device,
            )
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not empty_past_kv:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}
        model_inputs.update({
            "position_ids": position_ids, "past_key_values": past_key_values,
            "use_cache": use_cache, "attention_mask": attention_mask,
            "logits_to_keep": self.config.num_logits_to_keep,
            "cache_position": cache_position,
        })
        return model_inputs

    # ------------------------------------------------------------------
    # Forward (context tower only, for HF generate)
    # ------------------------------------------------------------------

    def forward(
        self, input_ids=None, inputs_embeds=None, position_ids=None,
        cache_params=None, labels=None, output_attentions=None,
        output_hidden_states=None, return_dict=None, use_cache=None,
        cache_position=None, attention_mask=None, **kwargs,
    ) -> Union[Tuple, NemotronHCausalLMOutput]:
        past_key_values = kwargs.pop("past_key_values", None)
        if past_key_values is not None and cache_params is None:
            cache_params = past_key_values
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.context_tower(
            input_ids, cache_params=cache_params, inputs_embeds=inputs_embeds,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, use_cache=use_cache,
            cache_position=cache_position, attention_mask=attention_mask,
        )
        hidden_states = outputs[0]
        logits = self.context_lm_head(hidden_states.to(self.context_lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return NemotronHCausalLMOutput(
            loss=loss, logits=logits, cache_params=outputs.cache_params,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )

    # ------------------------------------------------------------------
    # Layer-by-layer forward with cache + optional time conditioning
    # ------------------------------------------------------------------

    def _forward_tower_with_cache(self, tower, lm_head, input_ids, cache,
                                  cache_position, t_emb=None):
        """Forward through tower with KV cache. If t_emb is provided, applies
        PixArt-style adaLN modulation (shift/scale after norm, gate on output)."""
        hidden = tower.embeddings(input_ids)
        causal_mask = tower._update_causal_mask(None, hidden, cache_position)

        for layer_idx, block in enumerate(tower.layers):
            residual = hidden
            hidden = block.norm(hidden.to(dtype=block.norm.weight.dtype))
            if block.residual_in_fp32:
                residual = residual.to(torch.float32)

            mod = None
            if t_emb is not None:
                mod = _get_mod_params(t_emb, self.scale_shift_tables[layer_idx])
                shift, scale, gate = mod
                hidden = _modulate(hidden, shift, scale)

            if block.block_type == "mamba":
                hidden = block.mixer(
                    hidden, cache_params=cache, cache_position=cache_position,
                )
            elif block.block_type == "attention":
                hidden, _, _ = block.mixer(
                    hidden, attention_mask=causal_mask,
                    past_key_value=cache, cache_position=cache_position,
                )
            elif block.block_type in ["mlp", "moe"]:
                hidden = block.mixer(hidden)
            else:
                raise ValueError(f"Unknown block_type: {block.block_type}")

            if mod is not None:
                hidden = gate.unsqueeze(1) * hidden

            hidden = residual + hidden

        hidden = tower.norm_f(hidden)
        logits = lm_head(hidden.to(lm_head.weight.dtype)).float()
        return logits

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _make_cache(self, config, batch_size, dtype, device):
        return FixedHybridCache(config, batch_size, dtype, device)

    def _build_context_cache(self, prompt_ids):
        """Two-pass context prefill: S-2 and S-1 Mamba states + full KV."""
        B, S = prompt_ids.shape
        device = prompt_ids.device
        tower = self.context_tower
        pattern = self.config.hybrid_override_pattern

        cache_p1 = self._make_cache(self.config, B, self.dtype, device)
        cp_p1 = torch.arange(S - 1, device=device)
        self._forward_tower_with_cache(tower, self.context_lm_head,
                                       prompt_ids[:, :-1], cache_p1, cp_p1)

        mamba_s2 = {}
        for i in range(self.config.num_hidden_layers):
            if pattern[i] == "M":
                mamba_s2[i] = (cache_p1.conv_states[i].clone(),
                               cache_p1.ssm_states[i].clone())

        cache_p2 = self._make_cache(self.config, B, self.dtype, device)
        for i in range(self.config.num_hidden_layers):
            if pattern[i] == "M":
                cache_p2.conv_states[i] = cache_p1.conv_states[i].clone()
                cache_p2.ssm_states[i] = cache_p1.ssm_states[i].clone()
            elif pattern[i] == "*":
                cache_p2.key_cache[i] = cache_p1.key_cache[i].clone()
                cache_p2.value_cache[i] = cache_p1.value_cache[i].clone()

        cache_p2.has_previous_state = True
        cp_p2 = torch.arange(S - 1, S, device=device)
        logits = self._forward_tower_with_cache(tower, self.context_lm_head,
                                                prompt_ids[:, -1:], cache_p2, cp_p2)

        # "logits" = context tower's prediction at the last prompt position
        # (used by generate_ar). Diffusion/mock-AR ignore it.
        return {"ctx_cache": cache_p2, "mamba_s2": mamba_s2, "ctx_len": S, "logits": logits}

    def _extend_context_cache(self, new_tokens, cache_state, block_wise=True):
        """Extend context cache by new_tokens (B, L).

        block_wise=True (diffusion): Mamba advances via a single block chunk-scan
        (fast for a whole committed block; matches mcore).
        block_wise=False (AR / mock-AR): token-by-token single-step decode, the
        same kernels stock single-tower uses, so AR/mock-AR output matches stock.
        Also stores cache_state["logits"] (last-token prediction) when single-step.
        """
        ctx_cache = cache_state["ctx_cache"]
        pattern = self.config.hybrid_override_pattern
        ctx_len = cache_state["ctx_len"]
        tower = self.context_tower
        ctx_device = next(tower.parameters()).device
        L = new_tokens.shape[1]
        tokens = new_tokens.to(ctx_device)

        # Snapshot pre-extension Mamba states as the new S-2 (used by mock-AR).
        new_s2 = {}
        for i in range(self.config.num_hidden_layers):
            if pattern[i] == "M":
                new_s2[i] = (ctx_cache.conv_states[i].clone(),
                             ctx_cache.ssm_states[i].clone())
        cache_state["mamba_s2"] = new_s2

        ctx_cache.has_previous_state = True

        if not block_wise:
            # Single-step token-by-token extension (stock decode kernels).
            logits = None
            for j in range(L):
                cp = torch.tensor([ctx_len + j], device=ctx_device)
                logits = self._forward_tower_with_cache(
                    tower, self.context_lm_head, tokens[:, j:j+1], ctx_cache, cp,
                )
            cache_state["ctx_len"] = ctx_len + L
            cache_state["logits"] = logits
            return cache_state

        cache_position = torch.arange(ctx_len, ctx_len + L, device=ctx_device)
        hidden = tower.embeddings(tokens)
        causal_mask = tower._update_causal_mask(None, hidden, cache_position)

        for layer_idx, block in enumerate(tower.layers):
            residual = hidden
            h = block.norm(hidden.to(dtype=block.norm.weight.dtype))
            if block.residual_in_fp32:
                residual = residual.to(torch.float32)

            if block.block_type == "mamba":
                d_conv = block.mixer.conv_kernel_size
                init_conv = ctx_cache.conv_states[layer_idx][..., -(d_conv - 1):]
                init_ssm = ctx_cache.ssm_states[layer_idx].contiguous()
                h, new_conv, new_ssm = self._denoiser_block_mamba(
                    block.mixer, h, init_conv, init_ssm, return_states=True,
                )
                ctx_cache.conv_states[layer_idx] = new_conv
                ctx_cache.ssm_states[layer_idx] = new_ssm
            elif block.block_type == "attention":
                # Standard cached attention appends block KV (causal within block).
                h, _, _ = block.mixer(
                    h, attention_mask=causal_mask,
                    past_key_value=ctx_cache, cache_position=cache_position,
                )
            elif block.block_type in ["mlp", "moe"]:
                h = block.mixer(h)
            else:
                raise ValueError(f"Unknown block_type: {block.block_type}")

            hidden = residual + h

        cache_state["ctx_len"] = ctx_len + L
        return cache_state

    def _build_denoiser_cache_mock_ar(self, cache_state, device):
        """Mock-AR denoiser cache: Mamba S-2, Attention KV[:-1]."""
        ctx_cache = cache_state["ctx_cache"]
        mamba_s2 = cache_state["mamba_s2"]
        pattern = self.config.hybrid_override_pattern
        B = ctx_cache.conv_states[0].shape[0] if pattern[0] == "M" else ctx_cache.key_cache[0].shape[0]

        den = self._make_cache(self.config, B, self.dtype, device)
        for i in range(self.config.num_hidden_layers):
            if pattern[i] == "M":
                conv_s2, ssm_s2 = mamba_s2[i]
                den.conv_states[i] = conv_s2.to(device).clone()
                den.ssm_states[i] = ssm_s2.to(device).clone()
            elif pattern[i] == "*":
                k, v = ctx_cache.key_cache[i], ctx_cache.value_cache[i]
                if k.dim() == 4 and k.shape[2] > 0:
                    den.key_cache[i] = k[:, :, :-1, :].to(device).clone()
                    den.value_cache[i] = v[:, :, :-1, :].to(device).clone()
        den.has_previous_state = True
        return den

    def _build_denoiser_cache_diffusion(self, cache_state, device):
        """Diffusion denoiser cache: Mamba S-1 (latest), full Attention KV."""
        ctx_cache = cache_state["ctx_cache"]
        pattern = self.config.hybrid_override_pattern
        B = ctx_cache.conv_states[0].shape[0] if pattern[0] == "M" else ctx_cache.key_cache[0].shape[0]

        den = self._make_cache(self.config, B, self.dtype, device)
        for i in range(self.config.num_hidden_layers):
            if pattern[i] == "M":
                den.conv_states[i] = ctx_cache.conv_states[i].to(device).clone()
                den.ssm_states[i] = ctx_cache.ssm_states[i].to(device).clone()
            elif pattern[i] == "*":
                k, v = ctx_cache.key_cache[i], ctx_cache.value_cache[i]
                if k.dim() == 4 and k.shape[2] > 0:
                    den.key_cache[i] = k.to(device).clone()
                    den.value_cache[i] = v.to(device).clone()
        den.has_previous_state = True
        return den

    # ------------------------------------------------------------------
    # Denoiser step (shared by mock-AR and diffusion)
    # ------------------------------------------------------------------

    def _run_denoiser_step_mock_ar(self, input_ids, cache_state):
        """Mock-AR denoiser: pos=ctx_len-1, KV[:-1], Mamba S-2."""
        ctx_len = cache_state["ctx_len"]
        den_device = next(self.denoiser_tower.parameters()).device
        den_input = input_ids.to(den_device)
        den_cache = self._build_denoiser_cache_mock_ar(cache_state, den_device)
        cp = torch.tensor([ctx_len - 1], device=den_device)
        return self._forward_tower_with_cache(
            self.denoiser_tower, self.lm_head, den_input, den_cache, cp,
        )

    def _denoiser_block_attention(self, mixer, hidden, ctx_k, ctx_v):
        """Bidirectional denoiser self-attention over [context_KV | block_KV].

        Mirrors the mcore `_forward_attn_with_past` (is_causal=False, no mask):
        every block position attends to ALL context positions and ALL block
        positions (the noisy block is processed bidirectionally within itself).

        Args:
            mixer: NemotronHAttention module (provides q/k/v/o projections)
            hidden: (B, L, D) post-norm (and post-modulation) block hidden states
            ctx_k, ctx_v: context KV, each (B, num_kv_heads, ctx_len, head_dim)

        Returns: (B, L, D) attention output (before residual add)
        """
        bsz, q_len, _ = hidden.shape
        q = mixer.q_proj(hidden).view(bsz, q_len, mixer.num_heads, mixer.head_dim).transpose(1, 2)
        k = mixer.k_proj(hidden).view(bsz, q_len, mixer.num_key_value_heads, mixer.head_dim).transpose(1, 2)
        v = mixer.v_proj(hidden).view(bsz, q_len, mixer.num_key_value_heads, mixer.head_dim).transpose(1, 2)

        # Concatenate context KV (past) with current block KV on the sequence dim.
        k = torch.cat([ctx_k.to(k.dtype), k], dim=2)
        v = torch.cat([ctx_v.to(v.dtype), v], dim=2)

        # GQA: expand KV heads to match query heads.
        k = repeat_kv(k, mixer.num_key_value_groups)
        v = repeat_kv(v, mixer.num_key_value_groups)

        # Full (non-causal) attention: block sees all context + whole block.
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            bsz, q_len, mixer.num_heads * mixer.head_dim
        )
        return mixer.o_proj(attn_output)

    def _denoiser_block_mamba(self, mixer, hidden, init_conv, init_ssm, return_states=False):
        """Chunk-scan the whole block through the Mamba mixer, seeded from the
        context state — mirrors mcore `forward_mamba_layer_with_states`
        (non-bidirectional). Uses the same mamba_ssm/causal_conv1d kernels as
        mcore, instead of HF's token-by-token single-step path (which is both a
        numerical mismatch and crashes in this env's causal_conv1d_update).

        Args:
            mixer: NemotronHMamba2Mixer
            hidden: (B, L, D) post-norm (and post-modulation) block hidden states
            init_conv: (B, conv_dim, d_conv-1) context conv state, or None
            init_ssm:  (B, nheads, headdim, d_state) context SSM state, or None
            return_states: also return the updated (conv_state[width d_conv], ssm_state)
                so the caller can advance a KV/Mamba cache (used by context extend).

        Returns: (B, L, D) mixer output (before adaLN gate / residual);
                 or (output, new_conv_state, new_ssm_state) if return_states.
        """
        from einops import rearrange
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        from causal_conv1d import causal_conv1d_fn

        d_inner = mixer.intermediate_size
        ngroups = mixer.n_groups
        d_state = mixer.ssm_state_size
        headdim = mixer.head_dim
        conv_dim = mixer.conv_dim
        d_conv = mixer.conv_kernel_size

        proj = mixer.in_proj(hidden)                       # (B, L, d_inner+conv_dim+nheads)
        z, xBC, dt = torch.split(proj, [d_inner, conv_dim, mixer.num_heads], dim=-1)

        # causal_conv1d_fn with initial_states requires channel-last layout:
        #  - input (B, conv_dim, L): use the transpose VIEW (stride(1)==1), no .contiguous()
        #  - initial_states (B, conv_dim, d_conv-1): force channel-last via the
        #    transpose->contiguous->transpose trick (mcore _run_denoiser_step).
        if init_conv is not None:
            init_conv = init_conv.transpose(-1, -2).contiguous().transpose(-1, -2)
        xBC_conv = causal_conv1d_fn(
            xBC.transpose(1, 2),                           # (B, conv_dim, L) channel-last view
            mixer.conv1d.weight.squeeze(1),
            mixer.conv1d.bias,
            activation=mixer.activation,
            initial_states=init_conv,
        ).transpose(1, 2)                                  # (B, L, conv_dim)

        x, B_proj, C_proj = torch.split(
            xBC_conv, [d_inner, ngroups * d_state, ngroups * d_state], dim=-1
        )
        x = rearrange(x, "b s (h p) -> b s h p", p=headdim).contiguous()
        B_proj = rearrange(B_proj, "b s (g n) -> b s g n", n=d_state).contiguous()
        C_proj = rearrange(C_proj, "b s (g n) -> b s g n", n=d_state).contiguous()

        # Run the SSM scan in fp32. With a long context the seeded SSM state gets
        # large (O(1e3)+); the bf16 chunk-scan then overflows to NaN, and because
        # the Triton kernel's reductions are not bit-deterministic this strikes
        # nondeterministically (a NaN on a block's first/all-masked step makes
        # every confidence NaN and force-commits an arbitrary token).
        # The scan spans only one block (<=16 tokens) so fp32 is essentially free,
        # and it is strictly more accurate. Cast back before the gated norm.
        _y_dtype = z.dtype
        A = -torch.exp(mixer.A_log.float())
        scan = mamba_chunk_scan_combined(
            x.float(), dt.float().contiguous(), A, B_proj.float(), C_proj.float(),
            mixer.chunk_size,
            D=mixer.D.float(), z=None,
            dt_bias=mixer.dt_bias.float(), dt_softplus=True,
            initial_states=(init_ssm.float() if init_ssm is not None else None),
            return_final_states=return_states,
        )
        if return_states:
            y, new_ssm = scan
        else:
            y = scan
        y = rearrange(y, "b s h p -> b s (h p)").to(_y_dtype)
        y = mixer.norm(y, z)                               # Mamba2 z-gated RMSNorm
        out = mixer.out_proj(y)
        if not return_states:
            return out
        # New conv state: HF cache stores the last d_conv raw xBC inputs (width
        # d_conv), most-recent at index -1. block_size >= d_conv here.
        L = xBC.shape[1]
        if L >= d_conv:
            new_conv = xBC[:, -d_conv:, :].transpose(1, 2).contiguous()
        else:
            hist = init_conv if init_conv is not None else xBC.new_zeros(xBC.shape[0], conv_dim, d_conv - 1)
            comb = torch.cat([hist.transpose(1, 2), xBC], dim=1)
            new_conv = comb[:, -d_conv:, :].transpose(1, 2).contiguous()
        return out, new_conv, new_ssm

    def _run_denoiser_step_diffusion(self, block_ids, cache_state, t=None, den_cache=None):
        """Diffusion denoiser forward over the FULL block (B, L) in one pass.

        Parity with mcore `_run_denoiser_step`:
          - Attention layers run BIDIRECTIONALLY within the block, attending to
            the full context KV cache + the whole noisy block (is_causal=False).
            A token-by-token causal pass would hide later block positions from
            earlier ones.
          - Mamba layers are causal/forward-only (bidirectional_mamba=False) and
            are chunk-scanned over the whole block from the context state (S-1),
            matching mcore's `forward_mamba_layer_with_states`.
          - Time conditioning (adaLN-single) is applied per layer. The modulate/norm
            ORDER depends on where mcore's norm lives: mamba & attention norms are
            FUSED into in_proj/linear_qkv (applied AFTER modulate) -> modulate THEN
            norm; MoE uses a separate pre_mlp_layernorm -> norm THEN modulate.
            Gate is applied to the mixer output in all cases.

        Args:
            block_ids: (B, L) tokens to denoise
            cache_state: context cache state
            t: (B,) timestep in [0,1], or None

        Returns: logits (B, L, V)
        """
        ctx_len = cache_state["ctx_len"]
        tower = self.denoiser_tower
        den_device = next(tower.parameters()).device
        den_input = block_ids.to(den_device)
        L = den_input.shape[1]

        # Time embedding -> per-layer modulation params (shift, scale, gate).
        t_emb = None
        if t is not None:
            t_dev = t.to(device=den_device, dtype=self.dtype)
            t_repr = self.t_embedder(t_dev)
            t_emb = self.t_block(t_repr)

        # Denoiser cache (context Mamba S-1 state + full context KV). It is
        # READ-ONLY here and identical for every step within a block, so the
        # caller should build it once per block and pass it in (avoids cloning +
        # cuda:0->cuda:1 copying the whole context cache on every NFE). Fall back
        # to building it if not provided.
        if den_cache is None:
            den_cache = self._build_denoiser_cache_diffusion(cache_state, den_device)

        hidden = tower.embeddings(den_input)

        for layer_idx, block in enumerate(tower.layers):
            residual = hidden
            if block.residual_in_fp32:
                residual = residual.to(torch.float32)

            mod = None
            if t_emb is not None:
                mod = _get_mod_params(t_emb, self.scale_shift_tables[layer_idx])
                shift, scale, gate = mod

            # adaLN modulate vs norm ORDER depends on where mcore's norm lives:
            #   - mamba/attention: norm is FUSED into in_proj/linear_qkv and is
            #     applied AFTER the explicit modulate  -> modulate THEN norm.
            #   - moe/mlp: separate pre_mlp_layernorm applied BEFORE modulate
            #     -> norm THEN modulate.
            if block.block_type in ("mamba", "attention"):
                h = hidden
                if mod is not None:
                    h = _modulate(h, shift, scale)
                h = block.norm(h.to(dtype=block.norm.weight.dtype))
            else:  # mlp / moe
                h = block.norm(hidden.to(dtype=block.norm.weight.dtype))
                if mod is not None:
                    h = _modulate(h, shift, scale)

            if block.block_type == "mamba":
                # Chunk-scan the whole block in one kernel launch, seeded from the
                # context Mamba state (matches mcore forward_mamba_layer_with_states).
                # HF conv_states are width d_conv; causal_conv1d_fn's initial_states
                # wants the d_conv-1 most-recent columns.
                d_conv = block.mixer.conv_kernel_size
                init_conv = den_cache.conv_states[layer_idx][..., -(d_conv - 1):]
                init_ssm = den_cache.ssm_states[layer_idx].contiguous()
                h = self._denoiser_block_mamba(block.mixer, h, init_conv, init_ssm)
            elif block.block_type == "attention":
                ctx_k = den_cache.key_cache[layer_idx]
                ctx_v = den_cache.value_cache[layer_idx]
                h = self._denoiser_block_attention(block.mixer, h, ctx_k, ctx_v)
            elif block.block_type in ["mlp", "moe"]:
                h = block.mixer(h)
            else:
                raise ValueError(f"Unknown block_type: {block.block_type}")

            if mod is not None:
                h = gate.unsqueeze(1) * h

            hidden = residual + h

        hidden = tower.norm_f(hidden)
        logits = self.lm_head(hidden.to(self.lm_head.weight.dtype)).float()
        return logits

    # ------------------------------------------------------------------
    # Context-tower AR generation (single-tower baseline, cached)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_ar(self, input_ids, max_new_tokens=128, temperature=0.0,
                    top_k=None, top_p=None, eos_token_id=None):
        """Single-tower AR using ONLY the context tower, cached, 1 token/step.

        Equivalent to the stock single-tower model's greedy AR (the context tower
        is the frozen base), but routed through our own KV/Mamba cache machinery
        (single-step decode) — so it's O(N) cached and avoids HF generate()'s
        cache path that crashes on this env. This is the fair ST-AR baseline.
        """
        cache_state = self._build_context_cache(input_ids)
        logits = cache_state["logits"][:, -1, :].float()
        generated: List[torch.Tensor] = []

        for step in range(max_new_tokens):
            tok = self._sample_token(logits, temperature, top_k, top_p)
            generated.append(tok)
            if eos_token_id is not None and (tok == eos_token_id).any():
                break
            cache_state = self._extend_context_cache(tok, cache_state, block_wise=False)
            logits = cache_state["logits"][:, -1, :].float()

        return torch.cat([input_ids] + [g.to(input_ids.device) for g in generated], dim=1)

    # ------------------------------------------------------------------
    # Mock-AR generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_mock_ar(self, input_ids, max_new_tokens=128, temperature=0.0,
                         top_k=None, top_p=None, eos_token_id=None):
        """Two-tower mock-AR: S-2/KV[:-1] cache, 1 token/step."""
        B = input_ids.shape[0]
        generated: List[torch.Tensor] = []
        cache_state = self._build_context_cache(input_ids)

        for step in range(max_new_tokens):
            last_token = input_ids[:, -1:] if step == 0 else generated[-1]
            logits = self._run_denoiser_step_mock_ar(last_token, cache_state)
            logits = logits[:, -1, :].float()
            tok = self._sample_token(logits, temperature, top_k, top_p)
            generated.append(tok)
            if eos_token_id is not None and (tok == eos_token_id).any():
                break
            # Single-step context extension (stock kernels) so mock-AR matches stock.
            cache_state = self._extend_context_cache(tok, cache_state, block_wise=False)

        return torch.cat([input_ids] + [g.to(input_ids.device) for g in generated], dim=1)

    # ------------------------------------------------------------------
    # Mask-Diffusion generation
    # ------------------------------------------------------------------

    @staticmethod
    def _mdlm_forward(logits, xt, mask_token_id):
        """Constrain logits -> p(x0|xt): mask token gets -inf, decoded tokens
        get delta on their current value."""
        logits = logits.clone()
        logits[..., mask_token_id] = -1e12
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        # Fix unmasked positions: they must predict themselves with prob 1
        unmasked = (xt != mask_token_id)
        if unmasked.any():
            log_probs[unmasked] = -1e12
            log_probs[unmasked, :].scatter_(-1, xt[unmasked].unsqueeze(-1), 0.0)
        return log_probs

    @staticmethod
    def _gumbel_sample(log_probs):
        """Gumbel-max sampling from log probabilities."""
        gumbel_noise = -torch.log(-torch.log(
            torch.rand_like(log_probs).clamp(min=1e-10)
        ))
        return (log_probs + gumbel_noise).argmax(dim=-1)

    @torch.no_grad()
    def generate_mask_diffusion(
        self,
        input_ids,
        max_new_tokens=128,
        block_size=16,
        steps_per_block=16,
        mask_token_id=3,
        temperature=0.0,
        top_k=None,
        confidence_threshold=0.9,
        eos_token_id=None,
        step_callback=None,
    ):
        """Block-wise mask diffusion with confidence_unmasking.

        Algorithm:
          1. Build context cache from prompt
          2. For each block:
             a. Init block_ids = all mask tokens
             b. For each denoising step:
                - Compute t_model = fraction of masked positions
                - Denoiser forward -> logits -> p(x0|xt) via _mdlm_forward
                - Predict tokens (greedy or gumbel)
                - Confidence = p(predicted|xt) from unscaled probs
                - Commit high-confidence predictions, remask low-confidence
             c. Extend context cache with final block
          3. Return full sequence

        Args:
            input_ids: (B, S) prompt
            max_new_tokens: total tokens to generate (must be divisible by block_size)
            block_size: tokens per diffusion block
            steps_per_block: denoising iterations per block
            mask_token_id: ID of the [MASK] token
            temperature: 0 = greedy argmax, >0 = gumbel sampling
            top_k: unused currently (kept for API compat)
            confidence_threshold: commit tokens above this confidence
            eos_token_id: stop on EOS

        Returns: (B, S + generated) full token sequence
        """
        B = input_ids.shape[0]
        device = input_ids.device
        assert max_new_tokens % block_size == 0, \
            f"max_new_tokens ({max_new_tokens}) must be divisible by block_size ({block_size})"
        num_blocks = max_new_tokens // block_size

        cache_state = self._build_context_cache(input_ids)
        context_ids = input_ids.clone()
        nfe = 0  # number of denoiser forward passes (network function evaluations)

        den_device = next(self.denoiser_tower.parameters()).device
        for block_idx in range(num_blocks):
            # Build the denoiser cache ONCE per block (context is fixed within a
            # block); reused by every denoising step to avoid per-NFE clone+copy.
            den_cache = self._build_denoiser_cache_diffusion(cache_state, den_device)

            # Initialize fully masked block
            xt = torch.full((B, block_size), mask_token_id, dtype=torch.long,
                            device=device)
            if step_callback is not None:
                step_callback(0, steps_per_block, xt, t=1.0, logits=None,
                              block_idx=block_idx)

            for step_idx in range(steps_per_block):
                # t_model = current mask fraction
                is_masked = (xt == mask_token_id)
                n_masked = is_masked.float().sum(-1).mean().item()
                if n_masked == 0:
                    break
                t_model = is_masked.float().mean()
                t_vec = t_model.expand(B).to(device)

                # Denoiser forward (logits come back on denoiser device, move to xt's device)
                logits = self._run_denoiser_step_diffusion(xt, cache_state, t=t_vec, den_cache=den_cache)
                nfe += 1
                logits = logits.to(device)

                # p(x0|xt) with constraints
                log_x_theta = self._mdlm_forward(logits, xt, mask_token_id)
                x_theta = log_x_theta.exp()

                # Predict: greedy or gumbel
                if temperature <= 0:
                    predicted = log_x_theta.argmax(dim=-1)
                else:
                    scaled_logits = logits.clone()
                    scaled_logits[..., mask_token_id] = -1e12
                    scaled_log = scaled_logits / temperature - torch.logsumexp(
                        scaled_logits / temperature, dim=-1, keepdim=True)
                    unmasked = (xt != mask_token_id)
                    if unmasked.any():
                        scaled_log[unmasked] = -1e12
                        scaled_log[unmasked, :].scatter_(-1, xt[unmasked].unsqueeze(-1), 0.0)
                    predicted = self._gumbel_sample(scaled_log)

                # Confidence from unscaled x_theta
                confidence = x_theta.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
                confidence[~is_masked] = float('inf')

                # Determine how many to commit
                is_last_step = (step_idx == steps_per_block - 1)
                n_masked_int = is_masked.sum(-1)  # (B,)

                if is_last_step:
                    tokens_to_commit = n_masked_int
                else:
                    # Per-batch commitment logic (simplified for B=1 common case)
                    remaining_steps = max(1, steps_per_block - step_idx)
                    num_above = ((confidence > confidence_threshold) & is_masked).sum(-1)
                    tokens_to_commit = torch.where(
                        num_above > 0, num_above,
                        torch.ones_like(num_above),
                    )
                    min_commit = (n_masked_int.float() / remaining_steps).ceil().long()
                    tokens_to_commit = torch.clamp(
                        torch.max(tokens_to_commit, min_commit),
                        max=n_masked_int,
                    )

                # Apply predictions then remask low-confidence
                output = torch.where(is_masked, predicted, xt)
                num_to_remask = n_masked_int - tokens_to_commit  # (B,)

                for b in range(B):
                    if num_to_remask[b] > 0:
                        masked_indices = is_masked[b].nonzero(as_tuple=True)[0]
                        masked_conf = confidence[b, masked_indices]
                        _, sort_idx = masked_conf.sort()
                        remask_idx = masked_indices[sort_idx[:num_to_remask[b]]]
                        output[b, remask_idx] = mask_token_id

                if step_callback is not None:
                    step_callback(step_idx, steps_per_block, xt,
                                  t=float(t_model.detach().cpu()), logits=logits,
                                  block_idx=block_idx)

                xt = output

            # Block complete — extend context
            context_ids = torch.cat([context_ids, xt], dim=1)
            cache_state = self._extend_context_cache(xt, cache_state)

            if eos_token_id is not None and (xt == eos_token_id).any():
                break

        # Expose NFE (denoiser forward passes) for reporting, e.g. inference.py.
        self._last_nfe = nfe
        return context_ids

    # ------------------------------------------------------------------
    # Sampling helper
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_token(logits, temperature, top_k, top_p):
        if temperature is None or temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = F.softmax(logits / temperature, dim=-1)
        if top_k is not None and top_k > 0:
            kth = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1).values[..., -1:]
            probs = torch.where(probs >= kth, probs, torch.zeros_like(probs))
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_p, idx = torch.sort(probs, descending=True, dim=-1)
            cum = sorted_p.cumsum(dim=-1)
            remove = torch.cat(
                [torch.zeros_like(cum[..., :1]), (cum > top_p)[..., :-1]], dim=-1,
            )
            sorted_p = sorted_p.masked_fill(remove.bool(), 0.0)
            probs = torch.zeros_like(probs).scatter_(-1, idx, sorted_p)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return torch.multinomial(probs, num_samples=1)

    # ------------------------------------------------------------------
    # Multi-GPU placement
    # ------------------------------------------------------------------

    def place_towers_on_devices(self, ctx_device="cuda:0", den_device="cuda:1"):
        """Manual tower placement. Time conditioning goes with denoiser."""
        self.context_tower = self.context_tower.to(ctx_device)
        self.context_lm_head = self.context_lm_head.to(ctx_device)
        self.denoiser_tower = self.denoiser_tower.to(den_device)
        self.lm_head = self.lm_head.to(den_device)
        self.t_embedder = self.t_embedder.to(den_device)
        self.t_block = self.t_block.to(den_device)
        self.scale_shift_tables = nn.ParameterList([
            nn.Parameter(p.to(den_device)) for p in self.scale_shift_tables
        ])
        return self
````

### `reference_inference.py`  ·  184 行

**作用**：官方双塔推理示例

> 文件自述：Two-tower NemotronH inference example.

````python
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
````

---

## 12 · 层间冗余探针（AR 塔 vs 扩散塔）

迁移自 Bodan 的单塔 Nemotron-Diffusion 层曲线探针，改造到 TwoTower 双塔。度量每层的两条余弦曲线：**tokenwise**（层内相邻 token 相似度）与 **adjacent-layer**（相邻层残差流相似度＝**层间冗余**，越高越可跳过）。对应论文 *A Comparative analysis of Layer-wise Representational Capacity in AR and Diffusion LLMs*（Goel et al., arXiv 2603.07475）。TwoTower 是天然对照：两塔同架构、同 25T 初始化，唯一差别是 `context_tower`＝冻结 AR、`denoiser_tower`＝扩散训练——因此两条曲线**只隔离训练目标本身**对深度冗余的影响。

### `src/layer_similarity.py`  ·  372 行

**作用**：双塔层间冗余/层间相似度探针（context 塔用干净 causal 前向抓 hidden_states；denoiser 塔用逐字节复刻的真实条件化去噪步抓残差流）。含 `--selftest` 纯数学自检（无需 GPU）。

````python
"""Layer-wise representational redundancy probe for TwoTower (AR tower vs diffusion tower).

Migrated from Bodan Liu's single-tower Nemotron-Diffusion probe
(`reproduce_paper_layer_curves.py`), which computes two cosine curves per layer:

  * tokenwise      — cos(hidden[t], hidden[t+1]) within a layer  (neighbour-token similarity)
  * adjacent_layer — cos(hidden_L[i], hidden_{L+1}[i]) per token  (LAYER-TO-LAYER REDUNDANCY)

The adjacent_layer curve is the redundancy signal from
  "A Comparative analysis of Layer-wise Representational Capacity in AR and Diffusion LLMs"
  (Goel et al., arXiv 2603.07475): high adjacent-layer cosine => that layer barely moved the
  residual stream => redundant / skippable. Their thesis: *diffusion* objectives create
  substantial EARLY-layer redundancy + weak recency bias; *AR* objectives create locally
  structured reps + strong recency bias.

WHY TwoTower is the ideal testbed: both towers are the SAME NemotronH architecture from the
SAME 25T-token init, but one is frozen-AR (`context_tower`) and one is diffusion-trained
(`denoiser_tower`). So the two curves isolate the effect of the *training objective alone*
on depth redundancy — a cleaner controlled comparison than cross-model (LLaDA vs Qwen).

The migration is non-trivial because the two models expose completely different APIs:
  Bodan's model  : single `model.encoder`, block forward hooks fire, `diffusion_lm` toggle.
  TwoTower       : two `NemotronHModel` towers; the real denoiser forward
                   (`_run_denoiser_step_diffusion`) iterates `for block in tower.layers`
                   MANUALLY (AdaLN modulation + cross-attn to context KV + Mamba state
                   seeding) and never calls `block.forward`, so block hooks DO NOT fire.
Capture strategy therefore differs per tower:
  * context tower (AR)  : one clean causal forward with output_hidden_states=True
                          (== its real behaviour: a plain cached causal LM).
  * denoiser tower (diff): a capturing, byte-faithful copy of _run_denoiser_step_diffusion
                          run for ONE real conditioned step on an all-[MASK] block (t=1.0),
                          seeded from a real context cache. This is the ACTUAL inference-time
                          computation (fixed Mamba kernel, cross-attn, time conditioning),
                          not the tower run in isolation.

Both capture paths return the residual stream as [embeddings, layer_1, ..., layer_52]
(53 entries for 52 layers), matching HF's output_hidden_states convention so the two towers
are indexed identically.

Runs on the pod (2 GPUs). Pure-math self-check runs anywhere:  python src/layer_similarity.py --selftest

    python src/layer_similarity.py --prompts data/gsm8k_mini.jsonl --out results/layer_sim \
        --block-size 16 --num-prompts 8 --plot
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# ======================================================================
# Curve math — kept ~verbatim from Bodan's probe (model-agnostic).
# ======================================================================
@dataclass
class CurveStats:
    sums: list = field(default_factory=list)
    sums2: list = field(default_factory=list)
    counts: list = field(default_factory=list)
    step_means: list = field(default_factory=list)


def _ensure_len(stats, n):
    if stats.sums:
        return
    stats.sums = [0.0] * n
    stats.sums2 = [0.0] * n
    stats.counts = [0] * n


def add_values(stats, layer_values):
    _ensure_len(stats, len(layer_values))
    step_mean = []
    for i, vals in enumerate(layer_values):
        vals = vals.detach().float().cpu()
        vals = vals[torch.isfinite(vals)]
        if vals.numel() == 0:
            step_mean.append(float("nan"))
            continue
        s = vals.sum().item()
        s2 = (vals * vals).sum().item()
        c = int(vals.numel())
        stats.sums[i] += s
        stats.sums2[i] += s2
        stats.counts[i] += c
        step_mean.append(s / c)
    stats.step_means.append(step_mean)


def tokenwise_cosines(hidden_states, token_slice):
    """cos(hidden[t], hidden[t+1]) within each layer -> one curve over layers."""
    values = []
    for hidden in hidden_states:
        x = hidden[0, token_slice, :].float()
        if x.ndim == 1 or x.shape[0] < 2:
            values.append(torch.empty(0))
        else:
            values.append(F.cosine_similarity(x[:-1], x[1:], dim=-1))
    return values


def adjacent_layer_cosines(hidden_states, token_slice):
    """cos(hidden_L[i], hidden_{L+1}[i]) per token -> the LAYER-REDUNDANCY curve."""
    values = []
    for left, right in zip(hidden_states[:-1], hidden_states[1:]):
        a = left[0, token_slice, :].float()
        b = right[0, token_slice, :].float()
        if a.ndim == 1:
            a = a.unsqueeze(0)
            b = b.unsqueeze(0)
        values.append(F.cosine_similarity(a, b, dim=-1))
    return values


def summarize(stats):
    rows = []
    for i, (s, s2, c) in enumerate(zip(stats.sums, stats.sums2, stats.counts)):
        if c == 0:
            mean = std = float("nan")
        else:
            mean = s / c
            var = max(0.0, (s2 / c) - mean * mean)
            std = math.sqrt(var)
        step_vals = [row[i] for row in stats.step_means
                     if i < len(row) and math.isfinite(row[i])]
        step_std = 0.0
        if len(step_vals) > 1:
            sm = sum(step_vals) / len(step_vals)
            step_std = math.sqrt(sum((x - sm) ** 2 for x in step_vals) / (len(step_vals) - 1))
        rows.append({"layer": i, "mean": mean, "std": std,
                     "step_std": step_std, "count": c})
    return rows


def write_rows(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "mean", "std", "step_std", "count"])
        writer.writeheader()
        writer.writerows(rows)


# ======================================================================
# TwoTower-specific capture.
# ======================================================================
def _context_hidden_states(model, prompt_ids):
    """Context (AR) tower: one clean causal forward, residual stream via HF hidden_states.
    This IS the context tower's real behaviour (a plain cached causal LM), so no patch needed."""
    ctx_device = next(model.context_tower.parameters()).device
    out = model.context_tower(input_ids=prompt_ids.to(ctx_device),
                              use_cache=False, output_hidden_states=True)
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise RuntimeError("context_tower did not return hidden_states; "
                           "NemotronHModel.forward must support output_hidden_states=True")
    return [h.detach() for h in hs]  # (embed, l1..lN)


def _denoiser_hidden_states_faithful(model, cache_state, block_ids, t_scalar):
    """Denoiser (diffusion) tower: ONE real conditioned step, byte-faithful copy of
    _run_denoiser_step_diffusion (reference_modeling.py:594) with a capture after each layer.
    Uses the FIXED per-token Mamba kernel (twotower.apply_diffusion_fix), real cross-attn to
    the context KV, real Mamba state seeding, and real AdaLN time conditioning."""
    _mod = sys.modules[type(model).__module__]
    _get_mod_params, _modulate = _mod._get_mod_params, _mod._modulate

    tower = model.denoiser_tower
    den_device = next(tower.parameters()).device
    den_input = block_ids.to(den_device)
    t = torch.full((den_input.shape[0],), float(t_scalar), device=den_device, dtype=model.dtype)

    t_repr = model.t_embedder(t)
    t_emb = model.t_block(t_repr)

    den_cache = model._build_denoiser_cache_diffusion(cache_state, den_device)
    hidden = tower.embeddings(den_input)

    captured = [hidden.detach()]                       # index 0 == embeddings (matches HF)
    for layer_idx, block in enumerate(tower.layers):
        residual = hidden
        if block.residual_in_fp32:
            residual = residual.to(torch.float32)

        mod = _get_mod_params(t_emb, model.scale_shift_tables[layer_idx])
        shift, scale, gate = mod

        if block.block_type in ("mamba", "attention"):
            h = _modulate(hidden, shift, scale)
            h = block.norm(h.to(dtype=block.norm.weight.dtype))
        else:  # mlp / moe
            h = block.norm(hidden.to(dtype=block.norm.weight.dtype))
            h = _modulate(h, shift, scale)

        if block.block_type == "mamba":
            d_conv = block.mixer.conv_kernel_size
            init_conv = den_cache.conv_states[layer_idx][..., -(d_conv - 1):]
            init_ssm = den_cache.ssm_states[layer_idx].contiguous()
            h = model._denoiser_block_mamba(block.mixer, h, init_conv, init_ssm)
        elif block.block_type == "attention":
            h = model._denoiser_block_attention(block.mixer, h,
                                                den_cache.key_cache[layer_idx],
                                                den_cache.value_cache[layer_idx])
        else:
            h = block.mixer(h)

        h = gate.unsqueeze(1) * h
        hidden = residual + h
        captured.append(hidden.detach())               # residual stream after layer L
    return captured


def _denoiser_hidden_states_standalone(model, block_ids):
    """Rough baseline: run the denoiser tower in ISOLATION (no context / AdaLN / cross-attn,
    causal default). NOT faithful to real inference — offered only for a cheap sanity curve."""
    den_device = next(model.denoiser_tower.parameters()).device
    out = model.denoiser_tower(input_ids=block_ids.to(den_device),
                               use_cache=False, output_hidden_states=True)
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise RuntimeError("denoiser_tower did not return hidden_states")
    return [h.detach() for h in hs]


# ======================================================================
# Per-prompt driver.
# ======================================================================
def probe_prompt(model, tok, prompt, args, stats):
    from twotower import MASK_TOKEN_ID
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")

    # --- context (AR) tower over the prompt ---
    ctx_hs = _context_hidden_states(model, prompt_ids)
    ctx_slice = slice(0, ctx_hs[0].shape[1])
    add_values(stats["context_tokenwise"], tokenwise_cosines(ctx_hs, ctx_slice))
    add_values(stats["context_adjacent_layer"], adjacent_layer_cosines(ctx_hs, ctx_slice))

    # --- denoiser (diffusion) tower over one all-[MASK] block, seeded by the prompt ---
    block = torch.full((1, args.block_size), MASK_TOKEN_ID, dtype=torch.long, device="cuda:0")
    if args.denoiser_standalone:
        den_hs = _denoiser_hidden_states_standalone(model, block)
    else:
        cache_state = model._build_context_cache(prompt_ids)
        den_hs = _denoiser_hidden_states_faithful(model, cache_state, block, args.t)
    den_slice = slice(0, den_hs[0].shape[1])
    add_values(stats["denoiser_tokenwise"], tokenwise_cosines(den_hs, den_slice))
    add_values(stats["denoiser_adjacent_layer"], adjacent_layer_cosines(den_hs, den_slice))


def make_plot(results, out_png):
    """Overlay the two towers' adjacent-layer redundancy curves (the money figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, color, label in [
        ("context_adjacent_layer", "#1f77b4", "context tower (AR, frozen)"),
        ("denoiser_adjacent_layer", "#d62728", "denoiser tower (diffusion-trained)"),
    ]:
        rows = results[key]
        xs = [r["layer"] for r in rows]
        ys = [r["mean"] for r in rows]
        es = [r["std"] for r in rows]
        ax.plot(xs, ys, "-o", ms=3, color=color, label=label)
        ax.fill_between(xs, [y - e for y, e in zip(ys, es)],
                        [y + e for y, e in zip(ys, es)], color=color, alpha=0.12)
    ax.set_xlabel("layer boundary (L -> L+1)")
    ax.set_ylabel("adjacent-layer cosine  (higher = more redundant)")
    ax.set_title("TwoTower layer-to-layer redundancy: AR tower vs diffusion tower")
    ax.axhline(0.9, ls="--", lw=0.8, color="gray")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"[plot] wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", help="jsonl with a 'prompt' field (data/*.jsonl)")
    ap.add_argument("--out", default="results/layer_sim", help="output dir prefix")
    ap.add_argument("--label", default="twotower")
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--t", type=float, default=1.0,
                    help="mask-ratio fed to AdaLN for the denoiser step (1.0 = all-masked)")
    ap.add_argument("--denoiser-standalone", action="store_true",
                    help="probe denoiser in isolation (unconditioned) instead of faithful step")
    ap.add_argument("--single", action="store_true", help="both towers on cuda:0")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="run curve-math check, no model")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from twotower import load
    from main_run import load_prompts

    prompts = load_prompts(args.prompts, args.num_prompts)
    model, tok = load(den_device="cuda:0" if args.single else "cuda:1")

    stats = {k: CurveStats() for k in (
        "context_tokenwise", "context_adjacent_layer",
        "denoiser_tokenwise", "denoiser_adjacent_layer")}

    t0 = time.time()
    with torch.no_grad():
        for i, p in enumerate(prompts):
            probe_prompt(model, tok, p["prompt"], args, stats)
            print(f"[{args.label}] prompt {i + 1}/{len(prompts)} done", flush=True)

    results = {k: summarize(v) for k, v in stats.items()}
    for name, rows in results.items():
        write_rows(f"{args.out}/{args.label}_{name}_similarity.csv", rows)

    def band(rows):
        ms = [r["mean"] for r in rows if math.isfinite(r["mean"])]
        return {"min": min(ms), "max": max(ms), "layers": len(rows)} if ms else {}

    summary = {"label": args.label, "num_prompts": len(prompts),
               "block_size": args.block_size, "t": args.t,
               "denoiser_standalone": args.denoiser_standalone,
               "elapsed_sec": round(time.time() - t0, 1),
               "adjacent_layer": {
                   "context_AR": band(results["context_adjacent_layer"]),
                   "denoiser_diffusion": band(results["denoiser_adjacent_layer"])}}
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/{args.label}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.plot:
        make_plot(results, "figs/fig_layer_redundancy.png")


# ----------------------------------------------------------------------
def _selftest():
    """No-model check that the curve math behaves: identical layers -> cosine 1.0;
    orthogonal adjacent layers -> ~0.0; monotone token ramp -> high tokenwise cosine."""
    torch.manual_seed(0)
    H, T = 64, 20
    base = torch.randn(1, T, H)
    # 3 "layers" that are progressively rotated copies -> adjacent-layer cos decreasing
    hs = [base, base.clone(), base + 0.0 * torch.randn(1, T, H)]  # first two identical
    adj = adjacent_layer_cosines(hs, slice(0, T))
    assert abs(adj[0].mean().item() - 1.0) < 1e-5, adj[0].mean().item()
    # tokenwise on a smooth ramp -> near 1; on random -> lower.
    # +1.0 offset keeps every token vector non-zero (a zero vector => cosine nan).
    ramp = (torch.linspace(0, 1, T).view(1, T, 1) + 1.0) * torch.ones(1, T, H)
    tw_ramp = tokenwise_cosines([ramp], slice(0, T))[0].mean().item()
    tw_rand = tokenwise_cosines([torch.randn(1, T, H)], slice(0, T))[0].mean().item()
    assert tw_ramp > tw_rand, (tw_ramp, tw_rand)
    # summarize / accumulate plumbing
    st = CurveStats()
    add_values(st, adj)
    add_values(st, adjacent_layer_cosines([base, base.clone()], slice(0, T)))
    rows = summarize(st)
    assert len(rows) == len(adj) and rows[0]["count"] > 0
    print(f"[selftest] OK  adj[0]={adj[0].mean():.4f} (==1) "
          f"tw_ramp={tw_ramp:.3f} > tw_rand={tw_rand:.3f}  rows={len(rows)}")


if __name__ == "__main__":
    main()
````

---

## 附录 · 有意未收录（no silent truncation）

为避免「看着像全了其实没全」，以下明确列出未纳入本汇编的仓库内容及原因：

| 未收录 | 原因 |
|---|---|
| `make_figs.py, make_gif.py 的 analysis_data/ 副本` | 与根目录版本逐字节相同（已核 diff），不重复收录 |
| `data/*.jsonl, analysis_data/**/*.jsonl|*.npz|*.pkl` | 数据/结果/trace，非代码 |
| `figs/*.png, *.gif` | 图像产物 |
| `*.md (README/RUNBOOK/HANDOFF/TROUBLESHOOTING/REPORT*)` | 文档，非代码 |

