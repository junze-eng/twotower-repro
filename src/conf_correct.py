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
