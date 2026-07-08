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
