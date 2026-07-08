"""Shared loading + instrumentation helpers for the TwoTower experiments."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16"
MASK_TOKEN_ID = 3  # confirmed by the HF model card; smoke_test re-checks against the tokenizer


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load(ctx_device="cuda:0", den_device="cuda:1", ar_only=False):
    """Load the model and place the towers. ar_only puts a single tower on ctx_device."""
    tok = load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    if ar_only:
        model = model.to(ctx_device)
    else:
        # denoiser tower -> den_device, frozen context/AR tower -> ctx_device
        model.place_towers_on_devices(ctx_device, den_device)
    model.eval()
    return model, tok


# --- NFE (denoiser forward count) instrumentation ---------------------------
# generate_mask_diffusion records model._last_nfe. Reset before each generation
# so tokens_per_nfe is measured per-call, not cumulatively.
def reset_nfe(model):
    setattr(model, "_last_nfe", None)


def get_nfe(model):
    return getattr(model, "_last_nfe", None)


def nan_guard_callback(state=None):
    """A step_callback that flags NaN/Inf in the denoiser logits during generation.
    Usage: cb = nan_guard_callback(); model.generate_mask_diffusion(..., step_callback=cb)
           then inspect cb.hits afterwards.
    """
    class _CB:
        def __init__(self):
            self.hits = []  # (block_idx, step_idx)
        def __call__(self, step_idx, steps_per_block, xt, t, logits, block_idx):
            if logits is not None and not torch.isfinite(logits).all():
                self.hits.append((int(block_idx), int(step_idx)))
    return _CB()
