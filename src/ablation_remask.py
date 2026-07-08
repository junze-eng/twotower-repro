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
