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
