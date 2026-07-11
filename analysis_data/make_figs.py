"""Local report figures for the TwoTower reproduction — no GPU, reads the downloaded jsonl.

Scoring is degeneration-aware: a GSM8K answer counts only if it is correct AND the output is
not a short-cycle repetition ("edededed" / "155555" garbage). That distinction turns the
block-collapse from a hidden 90% into an honest 50% clean-correct. All figure labels are in
English; the dataviz reference palette (light surface) is used throughout.
"""
import json, re, os, pickle
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
VIOLET, RED, ORANGE = "#4a3aa7", "#e34948", "#eb6834"
INK, INK2, MUTED, GRID, SURF, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
GOOD, CRIT = "#0ca30c", "#d03b3b"
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs"); os.makedirs(FIGS, exist_ok=True)


def load(fn):
    out = []
    for l in open(os.path.join(HERE, fn), encoding="utf-8"):
        l = l.strip()
        if l:
            try: out.append(json.loads(l))
            except: pass
    return out


def degen_frac(s):
    s = s.rstrip(); n = len(s)
    if n < 20: return 0.0
    best = 0
    for p in range(1, 9):
        i = 0
        while i < n:
            j = i
            while j + p < n and s[j] == s[j + p]:
                j += 1
            if j > i and (j - i) + p >= 2 * p + 6:
                best = max(best, (j - i) + p)
            i = max(j, i + 1)
    return best / n


def gsm_ok(output, ref):
    if ref is None: return False
    seg = re.split(r"\n\s*Question:", output)[0]
    m = re.findall(r"####\s*\$?\s*([\-0-9][\d,\.]*)", seg) or re.findall(r"([\-0-9][\d,\.]*)", seg)
    if not m: return False
    norm = lambda x: str(x).replace(",", "").rstrip(".").lstrip("$")
    try: return abs(float(norm(m[0])) - float(norm(ref))) < 1e-6
    except: return norm(m[0]) == norm(ref)


def agg(fn):
    a = defaultdict(lambda: dict(n=0, ok=0, clean=0, nfe=0.0, tpn=0.0))
    for r in load(fn):
        k = r.get("config_key", "?"); d = a[k]; d["n"] += 1
        ok = gsm_ok(r.get("output", ""), r.get("reference"))
        deg = degen_frac(r.get("output", "")) > 0.25
        d["ok"] += ok; d["clean"] += (ok and not deg)
        d["nfe"] += r.get("nfe", 0) or 0; d["tpn"] += r.get("tokens_per_nfe", 0) or 0
    return a


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)


def save(fig, name):
    p = os.path.join(FIGS, name); fig.savefig(p, dpi=140, facecolor=SURF); plt.close(fig)
    print("wrote", name)


# ---- FIG 1: gamma -> tokens/NFE (Claim A: still autoregressive) ----
def fig_gamma():
    rows = load("e1.jsonl"); gammas = [0.5, 0.7, 0.8, 0.9, 0.95]
    g = defaultdict(list)
    for r in rows: g[(r["gamma"], r["steps"])].append(r["tokens_per_nfe"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6)); fig.patch.set_facecolor(SURF); style(ax)
    for T, col in [(4, BLUE), (8, AQUA), (16, VIOLET)]:
        y = [sum(g[(gm, T)]) / len(g[(gm, T)]) for gm in gammas]
        ax.plot(gammas, y, "-o", color=col, lw=2.4, ms=7, label=f"steps={T}")
        ax.annotate(f"steps={T}", (gammas[-1], y[-1]), textcoords="offset points",
                    xytext=(6, 0), color=col, fontsize=9, fontweight="bold", va="center")
    ax.axhline(1.0, color=MUTED, ls="--", lw=1.2)
    ax.annotate("AR = 1.0 token / forward", (0.5, 1.12), color=MUTED, fontsize=8.5)
    ax.annotate("higher γ  →  closer to autoregressive", (0.7, 5.4), color=INK2, fontsize=9.5)
    ax.set_xlabel("confidence threshold  γ", color=INK2, fontsize=10)
    ax.set_ylabel("tokens / NFE  (in-block parallelism)", color=INK2, fontsize=10)
    ax.set_title("Claim A — still autoregressive: parallelism decays toward 1", color=INK,
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right"); ax.set_xlim(0.46, 1.02)
    fig.tight_layout(); save(fig, "fig_gamma.png")


# ---- FIG 2: ablations (Claim B) ----
def fig_ablation():
    ra, da = agg("abl_remask.jsonl"), agg("abl_denoiser.jsonl")
    labels = ["baseline", "remask OFF\n(no iteration)", "seed OFF\n(no Mamba seeding)", "time frozen\n(AdaLN fixed)"]
    src = [ra["baseline"], ra["remask_OFF"], da["disable_seed"], da["freeze_time"]]
    clean = [s["clean"] / s["n"] * 100 for s in src]; nfe = [s["nfe"] / s["n"] for s in src]
    cols = [MUTED, RED, CRIT, BLUE]; x = list(range(4))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 4.7)); fig.patch.set_facecolor(SURF)
    for ax, vals, ttl, fmt in [(a1, clean, "Quality — clean-correct %", "{:.0f}%"),
                               (a2, nfe, "Compute — NFE (lower = faster)", "{:.0f}")]:
        style(ax); ax.bar(x, vals, color=cols, width=0.62, zorder=2)
        for xi, v in zip(x, vals):
            ax.annotate(fmt.format(v), (xi, v), textcoords="offset points", xytext=(0, 4),
                        ha="center", color=INK, fontsize=10, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.3)
        ax.set_title(ttl, color=INK, fontsize=11, fontweight="bold", loc="left")
    a1.set_ylim(0, 112)
    a2.annotate("no seeding →\nconfidence never clears γ\n→ crawls ~1 token/step",
                xy=(2, 243), xytext=(0.28, 188), color=CRIT, fontsize=8.3, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=CRIT, lw=1.2))
    fig.suptitle("Claim B — seeding load-bearing · iteration lifts quality · time-conditioning inert (freeze = no change)",
                 color=INK, fontsize=11.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94]); save(fig, "fig_ablation.png")


# ---- FIG 3: block collapse (Claim C) ----
def fig_collapse():
    a = agg("e3.jsonl"); blocks = [8, 16, 32, 64]
    key = lambda b: f"block_size{b}_gamma0.8_max_new256_steps16_temperature0.0"
    clean = [a[key(b)]["clean"] / a[key(b)]["n"] * 100 for b in blocks]
    naive = [a[key(b)]["ok"] / a[key(b)]["n"] * 100 for b in blocks]
    x = list(range(len(blocks)))
    fig, ax = plt.subplots(figsize=(7.2, 4.6)); fig.patch.set_facecolor(SURF); style(ax)
    ax.fill_between(x, clean, naive, color=RED, alpha=0.12)
    ax.plot(x, naive, "--o", color=MUTED, lw=2, ms=7, label="naive accuracy (first #### only)")
    ax.plot(x, clean, "-o", color=BLUE, lw=2.5, ms=9, label="clean-correct (correct & not degenerate)")
    for xi, c in zip(x, clean):
        ax.annotate(f"{c:.0f}%", (xi, c), textcoords="offset points", xytext=(0, 10),
                    ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.axvline(1, color=AXIS, lw=1, ls=":")
    ax.annotate("training block = 16", (1, 16), color=INK2, fontsize=8.5, ha="center")
    ax.annotate("block 64: half the outputs\ndegenerate into repetition", (3, 62), color=CRIT,
                fontsize=9, ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in blocks])
    ax.set_xlabel("sampling block size", color=INK2, fontsize=10)
    ax.set_ylabel("accuracy %", color=INK2, fontsize=10); ax.set_ylim(0, 108)
    ax.set_title("Claim C — block-size collapse hidden by naive accuracy", color=INK,
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="center left"); fig.tight_layout()
    save(fig, "fig_collapse.png")


# ---- FIG 4: quality-speed Pareto (e2) ----
def fig_pareto():
    a = agg("e2.jsonl")
    pts = []
    for k, d in a.items():
        g = float(re.search(r"gamma([\d.]+)", k).group(1))
        T = int(re.search(r"steps(\d+)", k).group(1))
        pts.append((d["tpn"] / d["n"], d["clean"] / d["n"] * 100, g, T))
    pts.sort()
    fig, ax = plt.subplots(figsize=(7.2, 4.6)); fig.patch.set_facecolor(SURF); style(ax)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, "-", color=GRID, lw=1.5, zorder=1)
    ax.scatter(xs, ys, color=BLUE, s=70, zorder=3)
    # per-point label offsets to defuse the top-left 100% cluster
    off = [(-8, 12, "right"), (0, 24, "center"), (10, 10, "left"),
           (-8, -15, "right"), (8, 9, "left"), (-8, -6, "right")]
    for (tpn, cc, g, T), (dx, dy, ha) in zip(pts, off):
        ax.annotate(f"γ={g}, T={T}", (tpn, cc), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, color=INK2, fontsize=8.2)
    ax.set_xlabel("tokens / NFE  →  faster (more parallel)", color=INK2, fontsize=10)
    ax.set_ylabel("clean-correct %", color=INK2, fontsize=10)
    ax.set_title("Quality – speed trade-off is mild (GSM8K-mini)", color=INK, fontsize=12,
                 fontweight="bold", loc="left")
    ax.set_ylim(min(ys) - 8, 107); ax.set_xlim(1.7, 7.6)
    fig.tight_layout(); save(fig, "fig_pareto.png")


# ---- FIG 5: steps per block (real trajectory signal) ----
def fig_stepsblock():
    tm = pickle.load(open(os.path.join(HERE, "trace_main.pkl"), "rb"))
    spbs = [{int(k): int(v) for k, v in tr["summary"]["steps_per_block"].items()} for tr in tm["traces"]]
    nb = max(max(s) for s in spbs) + 1
    mean = [sum(s.get(b, 0) for s in spbs) / len(spbs) for b in range(nb)]
    fig, ax = plt.subplots(figsize=(7.6, 4.4)); fig.patch.set_facecolor(SURF); style(ax)
    ax.bar(range(nb), mean, color=BLUE, width=0.7, zorder=2)
    ax.axhline(sum(mean) / len(mean), color=ORANGE, ls="--", lw=1.4)
    ax.annotate(f"mean {sum(mean)/len(mean):.1f}", (nb - 1.5, sum(mean) / len(mean) + 0.3),
                color=ORANGE, fontsize=9)
    ax.annotate("early blocks iterate more;\nlater blocks settle in ~2 steps\n(more context → higher confidence)",
                (5.5, 8.5), color=INK2, fontsize=9)
    ax.set_xticks(range(nb)); ax.set_xlabel("block index (generation order →)", color=INK2, fontsize=10)
    ax.set_ylabel("denoising steps used", color=INK2, fontsize=10)
    ax.set_title("Denoising steps per block (mean of 3 prompts) — measured", color=INK,
                 fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout(); save(fig, "fig_stepsblock.png")


# ---- FIG 6: MoE routing churn across denoising steps ----
def fig_moe():
    mo = pickle.load(open(os.path.join(HERE, "trace_moe.pkl"), "rb"))
    bylayer = defaultdict(list)
    for layer, step, arr in mo["moe_records"]:
        bylayer[layer].append((step, arr))
    ret = {}
    for layer, lst in bylayer.items():
        lst.sort(key=lambda x: x[0]); vals = []
        for (_, a0), (_, a1) in zip(lst, lst[1:]):
            for p in range(a0.shape[0]):
                vals.append(len(set(a0[p].tolist()) & set(a1[p].tolist())))
        if vals: ret[layer] = sum(vals) / len(vals)
    items = sorted(ret.items(), key=lambda kv: int(re.search(r"layers\.(\d+)", kv[0]).group(1)))
    lids = [int(re.search(r"layers\.(\d+)", k).group(1)) for k, _ in items]
    vals = [v for _, v in items]
    print("MoE retained experts/6 per layer:", {l: round(v, 2) for l, v in zip(lids, vals)})
    fig, ax = plt.subplots(figsize=(8.2, 4.4)); fig.patch.set_facecolor(SURF); style(ax)
    ax.bar([str(l) for l in lids], vals, color=VIOLET, width=0.7, zorder=2)
    ax.axhline(6 * 6 / 128, color=MUTED, ls="--", lw=1.2)
    ax.annotate("random-routing floor (≈0.28/6)", (0.2, 6 * 6 / 128 + 0.15), color=MUTED, fontsize=8.5)
    ax.set_ylim(0, 6)
    ax.set_xlabel("denoiser MoE layer index", color=INK2, fontsize=10)
    ax.set_ylabel("experts retained (of 6)\nbetween consecutive steps", color=INK2, fontsize=10)
    ax.set_title("MoE routing churns across denoising steps (a diffusion-only phenomenon)",
                 color=INK, fontsize=11.5, fontweight="bold", loc="left")
    fig.tight_layout(); save(fig, "fig_moe.png")


if __name__ == "__main__":
    fig_gamma(); fig_ablation(); fig_collapse(); fig_pareto(); fig_stepsblock(); fig_moe()
    print("done ->", FIGS)
