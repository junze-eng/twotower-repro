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
