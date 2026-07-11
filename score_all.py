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
