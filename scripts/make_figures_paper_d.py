"""
Paper D — figure generation.
Produces fig1_violin, fig2_calibration, fig3_efficiency.

Usage:
    python scripts/make_figures_paper_d.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats as sp

BASE    = Path(__file__).parent.parent
JSONL   = BASE / "kaggle_kernel/output_final2/e1_full_chr22.jsonl"
STATS   = BASE / "results/paper_d_stats_2026-08-23.json"
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# colour palette (colourblind-safe)
C_AFR = "#1f77b4"   # blue
C_EUR = "#d62728"   # red
C_LT  = "#aec7e8"   # light blue
C_LR  = "#f7b6b6"   # light red

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Load data ─────────────────────────────────────────────────────────────────
records = []
with open(JSONL) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("status") == "ok" and r.get("bias_score") is not None:
            records.append(r)

with open(STATS) as f:
    stats = json.load(f)

def get(model, vtype):
    return [r for r in records if r["model"] == model and r["vtype"] == vtype]

afr_h38 = get("NT-hg38",  "ref_minor")
eur_h38 = get("NT-hg38",  "control")
afr_1kg = get("NT-1000G", "ref_minor")
eur_1kg = get("NT-1000G", "control")

sc = lambda grp: np.array([r["bias_score"] for r in grp])

# ── Figure 1: Violin plot, 4 groups ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

groups = [
    (sc(afr_h38), C_AFR, "AFR\nNT-hg38"),
    (sc(eur_h38), C_EUR, "EUR\nNT-hg38"),
    (sc(afr_1kg), C_AFR, "AFR\nNT-1000G"),
    (sc(eur_1kg), C_EUR, "EUR\nNT-1000G"),
]

positions = [1, 2, 4, 5]
for (data, col, lbl), pos in zip(groups, positions):
    vp = ax.violinplot(data, positions=[pos], showmedians=False,
                       showextrema=False, widths=0.8)
    for body in vp["bodies"]:
        body.set_facecolor(col)
        body.set_alpha(0.45)
        body.set_edgecolor(col)
    # median + IQR box
    q25, q50, q75 = np.percentile(data, [25, 50, 75])
    ax.plot([pos, pos], [q25, q75], color=col, lw=4, solid_capstyle="round")
    ax.scatter([pos], [q50], color="white", s=30, zorder=5, linewidths=1.5, edgecolors=col)
    mean_val = np.mean(data)
    ax.scatter([pos], [mean_val], color=col, marker="D", s=40, zorder=6)

ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)

# significance brackets
def bracket(ax, x1, x2, y, text):
    h = 0.12
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color="black", lw=1.0)
    ax.text((x1+x2)/2, y+h+0.05, text, ha="center", va="bottom", fontsize=9)

p1 = stats["between_AFR_EUR_NT_hg38"]["p"]
p2 = stats["between_AFR_EUR_NT_1000G"]["p"]
bracket(ax, 1, 2, 7.5, f"p={p1:.1e}")
bracket(ax, 4, 5, 7.5, f"p={p2:.1e}")

# between-model reduction (AFR only)
p3 = stats["between_hg38_1000G_AFR"]["p"]
bracket(ax, 1, 4, 9.2, f"p={p3:.2f}")

ax.set_xticks(positions)
ax.set_xticklabels([lbl for _, _, lbl in groups], fontsize=10)
ax.set_ylabel("Bias score  [log P(ref k-mer) - log P(alt k-mer)]", fontsize=10)
ax.set_title("Reference allele bias by ancestry group and model", fontsize=12, fontweight="bold")
ax.set_xlim(0, 6)
ax.set_ylim(-9, 12.5)

# Annotate bootstrap CI + balanced resampling result
ci_lo = stats["bootstrap_ci_1000G"]["ci95_lo"]
ci_hi = stats["bootstrap_ci_1000G"]["ci95_hi"]
ax.text(4.5, -7.5,
        f"EUR−AFR gap (NT-1000G):\n"
        f"Δ={stats['bootstrap_ci_1000G']['mean_diff']:+.3f}, 95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]\n"
        f"Balanced resample (N=509 vs 509):\n"
        f"d=0.190, gap>0 in 100% of 1,000 iterations",
        fontsize=8, ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9e6", edgecolor="#e0a000", alpha=0.9))

legend_handles = [
    mpatches.Patch(color=C_AFR, alpha=0.6, label="African-enriched variants (N=7,684)"),
    mpatches.Patch(color=C_EUR, alpha=0.6, label="European-enriched variants (N=509)"),
    Line2D([0], [0], color="black", lw=1, ls="--", label="Zero bias (unbiased)"),
    Line2D([0], [0], marker="D", color="gray", markersize=7, ls="", label="Group mean"),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_violin.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved fig1_violin.png")

# ── Figure 2: Calibration scatter (2 panels) ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, model, afr, eur, title in [
    (axes[0], "NT-hg38",  afr_h38, eur_h38, "NT-hg38 (reference-only training)"),
    (axes[1], "NT-1000G", afr_1kg, eur_1kg, "NT-1000G (diverse training)"),
]:
    short = "hg38" if "hg38" in model else "1000G"

    sc_afr = np.array([r["bias_score"] for r in afr])
    af_afr = np.array([r["af_afr"] for r in afr])
    sc_eur = np.array([r["bias_score"] for r in eur])
    af_eur = np.array([r["af_eur"] for r in eur])

    # subsample AFR for readability
    rng = np.random.default_rng(42)
    idx = rng.choice(len(sc_afr), size=min(1500, len(sc_afr)), replace=False)
    ax.scatter(af_afr[idx], sc_afr[idx], c=C_AFR, alpha=0.15, s=8, linewidths=0, label="AFR")
    ax.scatter(af_eur,      sc_eur,      c=C_EUR, alpha=0.30, s=12, linewidths=0, label="EUR")

    # regression lines
    for af_arr, sc_arr, col in [(af_afr, sc_afr, C_AFR), (af_eur, sc_eur, C_EUR)]:
        slope, intercept, r_val, p_val, _ = sp.linregress(af_arr, sc_arr)
        x_line = np.linspace(af_arr.min(), af_arr.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color=col, lw=2.0, alpha=0.9)

    r_afr = stats[f"calibration_AFR_{short}"]["pearson_r"]
    p_afr = stats[f"calibration_AFR_{short}"]["pearson_p"]
    r_eur = stats[f"calibration_EUR_{short}"]["pearson_r"]
    p_eur = stats[f"calibration_EUR_{short}"]["pearson_p"]

    ax.text(0.02, 0.97,
            f"AFR: r={r_afr:+.3f} (p={p_afr:.2e})\nEUR: r={r_eur:+.3f} (p={p_eur:.2e})",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Within-population allele frequency (AF_pop)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

axes[0].set_ylabel("Bias score", fontsize=10)
fig.suptitle("Bias score calibration vs. within-population allele frequency", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_calibration.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved fig2_calibration.png")

# ── Figure 3: Ranking displacement — cumulative representation ────────────────
# Pool all variants by model, rank by bias score, show AFR/EUR cumulative %

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, model, afr, eur, title in [
    (axes[0], "NT-hg38",  afr_h38, eur_h38, "NT-hg38"),
    (axes[1], "NT-1000G", afr_1kg, eur_1kg, "NT-1000G"),
]:
    scores_afr = [(r["bias_score"], "AFR") for r in afr]
    scores_eur = [(r["bias_score"], "EUR") for r in eur]
    all_variants = sorted(scores_afr + scores_eur, key=lambda x: x[0], reverse=True)
    N = len(all_variants)
    n_afr = len(scores_afr)
    n_eur = len(scores_eur)

    # cumulative fraction of each group as we descend through ranked list
    afr_seen, eur_seen = 0, 0
    pct_thresholds = list(range(1, 101))
    afr_cum_obs = []   # observed fraction of top-k that is AFR
    eur_cum_obs = []
    afr_cum_exp = []   # expected (proportional baseline)
    eur_cum_exp = []

    for pct in pct_thresholds:
        k = max(1, int(N * pct / 100))
        a = sum(1 for _, g in all_variants[:k] if g == "AFR")
        e = k - a
        afr_cum_obs.append(100 * a / n_afr)
        eur_cum_obs.append(100 * e / n_eur)
        afr_cum_exp.append(pct)   # if perfectly proportional, exactly pct% of each group
        eur_cum_exp.append(pct)

    xs = pct_thresholds
    ax.plot(xs, afr_cum_obs, color=C_AFR, lw=2.2, label="AFR observed")
    ax.plot(xs, eur_cum_obs, color=C_EUR, lw=2.2, label="EUR observed")
    ax.plot(xs, afr_cum_exp, color="gray", lw=1.2, ls="--", alpha=0.7, label="Expected (proportional)")

    # shade divergence
    ax.fill_between(xs, afr_cum_obs, afr_cum_exp,
                    where=[o < e for o, e in zip(afr_cum_obs, afr_cum_exp)],
                    color=C_AFR, alpha=0.12, label="AFR under-represented")
    ax.fill_between(xs, eur_cum_obs, eur_cum_exp,
                    where=[o > e for o, e in zip(eur_cum_obs, eur_cum_exp)],
                    color=C_EUR, alpha=0.12, label="EUR over-represented")

    # annotate top-10% displacement
    top10_afr_obs = afr_cum_obs[9]
    top10_eur_obs = eur_cum_obs[9]
    ax.annotate(f"Top 10%:\nEUR {top10_eur_obs:.1f}% vs\nAFR {top10_afr_obs:.1f}%",
                xy=(10, top10_eur_obs), xytext=(25, top10_eur_obs + 8),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                fontsize=8.5, ha="left")

    ax.set_xlabel("Top-k% of variants ranked by bias score (high → low)", fontsize=10)
    ax.set_ylabel("Cumulative % of group captured", fontsize=10)
    ax.set_title(f"Ranking displacement — {title}", fontsize=11, fontweight="bold")
    ax.plot([0, 100], [0, 100], color="gray", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8.5, loc="lower right")

fig.suptitle(
    "EUR-enriched variants are systematically over-represented at high bias scores",
    fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_ranking_displacement.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved fig3_ranking_displacement.png")

print("\nAll figures saved to:", FIG_DIR)
