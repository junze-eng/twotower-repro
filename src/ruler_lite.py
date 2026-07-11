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
