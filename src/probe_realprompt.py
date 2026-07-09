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
