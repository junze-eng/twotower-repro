"""Block-diffusion denoising GIF, built from the REAL per-block step counts we still have.

HONEST SCOPE: block order and each block's denoising-step count come from the measured
trace (`steps_per_block` in trace_main.pkl). The exact position that commits at each step
was NOT captured (that capture bug is what we fixed but can't re-run without a pod), so the
WITHIN-block fill order is illustrative (left-to-right). The thing the animation actually
demonstrates from data: generation sweeps block-by-block (still left-to-right), and early
blocks take many more steps than later ones.
"""
import pickle, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans"]
BLUE, GREEN, ORANGE = (0.165, 0.470, 0.839), (0.0, 0.62, 0.31), (0.92, 0.41, 0.20)
MASK = (0.91, 0.91, 0.90)
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs"); os.makedirs(FIGS, exist_ok=True)

BS = 16
tm = pickle.load(open(os.path.join(HERE, "trace_main.pkl"), "rb"))
tr = tm["traces"][0]
spb = tr["summary"]["steps_per_block"]          # {block: n_steps}  -- REAL
spb = {int(k): int(v) for k, v in spb.items()}
NB = len(spb)                                    # 16 blocks

# --- precompute the global frame at which each (block,pos) cell commits ---
commit_frame = np.full((NB, BS), -1, np.int32)
f = 0
for b in range(NB):
    S = max(1, spb[b]); prev = 0
    for s in range(S):
        filled = round(BS * (s + 1) / S)         # linear within-block fill (illustrative)
        commit_frame[b, prev:filled] = f
        prev = filled; f += 1
    commit_frame[b, prev:] = f - 1
TOTAL = f

def color_for(age):
    if age < 0: return MASK
    if age == 0: return GREEN                    # committed this step
    if age <= 2: return ORANGE                   # recently committed
    return BLUE                                  # settled

frames = []
for fr in range(TOTAL):
    img = np.zeros((NB, BS, 3))
    for b in range(NB):
        for c in range(BS):
            cf = commit_frame[b, c]
            img[b, c] = color_for(fr - cf if (cf >= 0 and cf <= fr) else -1)
    cur_b = int(np.searchsorted(np.cumsum([max(1, spb[b]) for b in range(NB)]), fr, "right"))
    committed = int(((commit_frame >= 0) & (commit_frame <= fr)).sum())

    fig, ax = plt.subplots(figsize=(6.4, 6.8)); fig.patch.set_facecolor(SURF)
    ax.imshow(img, interpolation="nearest", aspect="equal", extent=[0, BS, NB, 0])
    for k in range(NB + 1): ax.axhline(k, color=SURF, lw=1.5)
    for k in range(BS + 1): ax.axvline(k, color=SURF, lw=1.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Block-diffusion denoising  (256-token reply)", color=INK, fontsize=13,
                 fontweight="bold", loc="left")
    ax.text(0, -0.6, "each ROW = one 16-token block · sweep is top→bottom (still left-to-right)",
            color=MUTED, fontsize=8.5)
    ax.set_xlabel(f"block {min(cur_b, NB-1)+1}/{NB}   ·   frame {fr+1}/{TOTAL}   ·   "
                  f"committed {committed}/{NB*BS}", color=INK2, fontsize=10)
    # legend
    for i, (col, lab) in enumerate([(GREEN, "just committed"), (ORANGE, "recent"),
                                    (BLUE, "settled"), (MASK, "masked")]):
        ax.add_patch(plt.Rectangle((i * 4.2, NB + 0.5), 0.7, 0.7, color=col, clip_on=False))
        ax.text(i * 4.2 + 0.9, NB + 1.15, lab, fontsize=8, color=INK2, clip_on=False)
    fig.tight_layout()
    p = os.path.join(FIGS, f"_gframe_{fr:03d}.png")
    fig.savefig(p, dpi=90, facecolor=SURF); plt.close(fig)
    frames.append(imageio.imread(p))

out = os.path.join(FIGS, "denoising.gif")
imageio.mimsave(out, frames + [frames[-1]] * 8, fps=6, loop=0)
# keep two representative stills for the report; drop the scratch frames
mid = TOTAL * 3 // 5
os.replace(os.path.join(FIGS, f"_gframe_{mid:03d}.png"), os.path.join(FIGS, "gif_still_mid.png"))
for fr in range(TOTAL):
    fp = os.path.join(FIGS, f"_gframe_{fr:03d}.png")
    if os.path.exists(fp): os.remove(fp)
print(f"wrote {out}  ({TOTAL} frames)  | steps_per_block sum={sum(spb.values())}")
print("still ->", os.path.join(FIGS, "gif_still_mid.png"))
