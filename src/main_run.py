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
