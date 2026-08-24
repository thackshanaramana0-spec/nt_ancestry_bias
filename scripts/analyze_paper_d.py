"""
Paper D — core statistical analysis.
Reads e1_full_chr22.jsonl, computes all stats for the paper, saves results JSON.

Usage:
    python scripts/analyze_paper_d.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp

BASE = Path(__file__).parent.parent
JSONL = BASE / "kaggle_kernel/output_final2/e1_full_chr22.jsonl"
OUT   = BASE / "results/paper_d_stats_2026-08-23.json"

OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
records = []
with open(JSONL) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("status") == "ok" and r.get("bias_score") is not None:
            records.append(r)

print(f"Loaded {len(records)} OK rows")

def get(model, vtype):
    return [r for r in records if r["model"] == model and r["vtype"] == vtype]

afr_h38 = get("NT-hg38",  "ref_minor")
eur_h38 = get("NT-hg38",  "control")
afr_1kg = get("NT-1000G", "ref_minor")
eur_1kg = get("NT-1000G", "control")

stats = {}

# ── 1. Descriptive statistics ─────────────────────────────────────────────────
print("\n=== 1. Descriptive statistics ===")
for label, grp, pop_af_key in [
    ("AFR_hg38",  afr_h38, "af_afr"),
    ("EUR_hg38",  eur_h38, "af_eur"),
    ("AFR_1000G", afr_1kg, "af_afr"),
    ("EUR_1000G", eur_1kg, "af_eur"),
]:
    sc = np.array([r["bias_score"] for r in grp])
    af = np.array([r[pop_af_key] for r in grp])
    t1, p1 = sp.ttest_1samp(sc, 0)
    pct_pos = float(np.mean(sc > 0))
    entry = {
        "n":        int(len(sc)),
        "mean":     float(np.mean(sc)),
        "sd":       float(np.std(sc)),
        "median":   float(np.median(sc)),
        "pct_pos":  float(pct_pos),
        "t_vs0":    float(t1),
        "p_vs0":    float(p1),
        "af_mean":  float(np.mean(af)),
        "af_median":float(np.median(af)),
    }
    stats[label] = entry
    print(f"{label}: N={entry['n']} mean={entry['mean']:+.4f} sd={entry['sd']:.4f} "
          f"%pos={100*pct_pos:.1f}% p_vs0={p1:.2e}")

# ── 2. Between-group: AFR vs EUR (same model) ─────────────────────────────────
print("\n=== 2. AFR vs EUR (between-group, same model) ===")
for model, afr, eur in [("NT-hg38", afr_h38, eur_h38), ("NT-1000G", afr_1kg, eur_1kg)]:
    sc_a = np.array([r["bias_score"] for r in afr])
    sc_e = np.array([r["bias_score"] for r in eur])
    t, p = sp.ttest_ind(sc_a, sc_e, equal_var=False)
    diff = float(np.mean(sc_e) - np.mean(sc_a))
    d    = diff / float(np.sqrt((np.var(sc_a) + np.var(sc_e)) / 2))
    key  = f"between_AFR_EUR_{model.replace('-','_')}"
    stats[key] = {"diff_EUR_minus_AFR": diff, "t": float(t), "p": float(p), "cohens_d": float(d)}
    print(f"{model}: EUR-AFR={diff:+.4f} t={t:.3f} p={p:.3e} d={d:.4f}")

# ── 3. Between-model: hg38 vs 1000G (same group) ─────────────────────────────
print("\n=== 3. hg38 vs 1000G (between-model, same group) ===")
for vtype, h38, kg in [("AFR", afr_h38, afr_1kg), ("EUR", eur_h38, eur_1kg)]:
    sc_h = np.array([r["bias_score"] for r in h38])
    sc_k = np.array([r["bias_score"] for r in kg])
    t, p = sp.ttest_ind(sc_h, sc_k, equal_var=False)
    diff = float(np.mean(sc_h) - np.mean(sc_k))
    d    = diff / float(np.sqrt((np.var(sc_h) + np.var(sc_k)) / 2))
    pct_red = diff / float(np.mean(sc_h)) * 100 if np.mean(sc_h) > 0 else float("nan")
    key = f"between_hg38_1000G_{vtype}"
    stats[key] = {"reduction": float(pct_red), "diff": diff, "t": float(t), "p": float(p), "cohens_d": float(d)}
    print(f"{vtype}: hg38-1000G={diff:+.4f} reduction={pct_red:.1f}% t={t:.3f} p={p:.3e} d={d:.4f}")

# ── 4. Within-group calibration (corr bias vs pop-AF) ─────────────────────────
print("\n=== 4. Within-group calibration ===")
for label, grp, ak in [
    ("AFR hg38",  afr_h38, "af_afr"),
    ("AFR 1000G", afr_1kg, "af_afr"),
    ("EUR hg38",  eur_h38, "af_eur"),
    ("EUR 1000G", eur_1kg, "af_eur"),
]:
    sc = np.array([r["bias_score"] for r in grp])
    af = np.array([r[ak] for r in grp])
    r_p, p_p = sp.pearsonr(af, sc)
    r_s, p_s = sp.spearmanr(af, sc)
    key = f"calibration_{label.replace(' ','_')}"
    stats[key] = {"pearson_r": float(r_p), "pearson_p": float(p_p),
                  "spearman_r": float(r_s), "spearman_p": float(p_s)}
    print(f"{label}: pearson_r={r_p:+.4f} p={p_p:.3e} | spearman_r={r_s:+.4f} p={p_s:.3e}")

# ── 5. Fisher z-test: compare AFR vs EUR within-group correlations ─────────────
print("\n=== 5. Fisher z-test (AFR corr vs EUR corr) ===")
for model in ["NT-hg38", "NT-1000G"]:
    short = "hg38" if "hg38" in model else "1000G"
    stat_tag = model.replace("-", "_")
    r_a = stats[f"calibration_AFR_{short}"]["pearson_r"]
    r_e = stats[f"calibration_EUR_{short}"]["pearson_r"]
    n_a = stats[f"AFR_{short}"]["n"]
    n_e = stats[f"EUR_{short}"]["n"]
    z_a = np.arctanh(r_a); z_e = np.arctanh(r_e)
    z_diff = (z_a - z_e) / np.sqrt(1/(n_a-3) + 1/(n_e-3))
    p_fish = float(2 * sp.norm.sf(abs(z_diff)))
    stats[f"fisher_z_{stat_tag}"] = {"r_AFR": r_a, "r_EUR": r_e,
                                      "z_diff": float(z_diff), "p": p_fish}
    print(f"{model}: r_AFR={r_a:+.4f} r_EUR={r_e:+.4f} z_diff={z_diff:.3f} p={p_fish:.3e}")

# ── 6. Calibration efficiency (NT-1000G, training-frequency normalised) ────────
print("\n=== 6. Calibration efficiency (NT-1000G) ===")
# 1000G composition: AFR=26%, EUR=20% (other pops not in our AF columns)
W_AFR = 0.26
W_EUR = 0.20

for label, grp, af_key_pop, af_key_other, w_pop, w_other in [
    ("AFR_1000G", afr_1kg, "af_afr", "af_eur", W_AFR, W_EUR),
    ("EUR_1000G", eur_1kg, "af_eur", "af_afr", W_EUR, W_AFR),
]:
    sc  = np.array([r["bias_score"] for r in grp])
    af_pop   = np.array([r[af_key_pop]   for r in grp])
    af_other = np.array([r[af_key_other] for r in grp])
    train_af = w_pop * af_pop + w_other * af_other
    valid = (train_af > 0.005) & (train_af < 0.995)
    expected = np.log((1 - train_af[valid]) / train_af[valid])
    eff = sc[valid] / expected
    entry = {
        "train_af_mean": float(np.mean(train_af)),
        "expected_bias_mean": float(np.mean(expected)),
        "observed_bias_mean": float(np.mean(sc)),
        "efficiency_mean": float(np.mean(eff)),
        "efficiency_pct": float(100 * np.mean(eff)),
        "n_valid": int(np.sum(valid)),
    }
    stats[f"efficiency_{label}"] = entry
    print(f"{label}: train_AF={entry['train_af_mean']:.4f} expected={entry['expected_bias_mean']:.3f} "
          f"obs={entry['observed_bias_mean']:+.4f} eff={entry['efficiency_pct']:.1f}%")

# ── 7. AF-stratified robustness check ─────────────────────────────────────────
print("\n=== 7. AF-stratified analysis ===")
bins = [(0.10, 0.20), (0.20, 0.35), (0.35, 1.01)]
stats["af_strat"] = {}
for lo, hi in bins:
    grp_a = [r for r in afr_1kg if lo <= r["af_afr"] < hi]
    grp_e = [r for r in eur_1kg if lo <= r["af_eur"] < hi]
    if not grp_a or not grp_e:
        print(f"  [{lo:.2f},{hi:.2f}): n_AFR={len(grp_a)} n_EUR={len(grp_e)} — skip")
        continue
    sc_a = np.array([r["bias_score"] for r in grp_a])
    sc_e = np.array([r["bias_score"] for r in grp_e])
    t, p = sp.ttest_ind(sc_a, sc_e, equal_var=False)
    diff = float(np.mean(sc_e) - np.mean(sc_a))
    key  = f"{lo:.2f}_{hi:.2f}"
    stats["af_strat"][key] = {
        "n_AFR": len(grp_a), "n_EUR": len(grp_e),
        "AFR_mean": float(np.mean(sc_a)), "EUR_mean": float(np.mean(sc_e)),
        "diff": diff, "p": float(p),
    }
    print(f"  [{lo:.2f},{hi:.2f}): n_AFR={len(grp_a)} n_EUR={len(grp_e)} "
          f"AFR={np.mean(sc_a):+.3f} EUR={np.mean(sc_e):+.3f} diff={diff:+.3f} p={p:.3e}")

# ── 8. Ranking displacement analysis (NT-1000G) ───────────────────────────────
print("\n=== 8. Ranking displacement (NT-1000G) ===")
sc_a = np.array([r["bias_score"] for r in afr_1kg])
sc_e = np.array([r["bias_score"] for r in eur_1kg])

afr_median_pct_in_eur = float(sp.percentileofscore(sc_e, np.median(sc_a)))
eur_75 = float(np.percentile(sc_e, 75))
eur_25 = float(np.percentile(sc_e, 25))
frac_afr_above_eur75 = float(np.mean(sc_a > eur_75))
frac_afr_below_eur25 = float(np.mean(sc_a < eur_25))

# Negative bias: model prefers ALT (common African alt flagged as unusual)
pct_afr_neg = float(np.mean(sc_a < 0))
pct_eur_neg = float(np.mean(sc_e < 0))
or_neg = (pct_afr_neg / (1 - pct_afr_neg)) / (pct_eur_neg / (1 - pct_eur_neg))
from scipy.stats import chi2_contingency
table = [[int(np.sum(sc_a < 0)), int(np.sum(sc_a >= 0))],
         [int(np.sum(sc_e < 0)), int(np.sum(sc_e >= 0))]]
chi2, p_chi, _, _ = chi2_contingency(table)

stats["ranking_displacement"] = {
    "afr_median_pct_in_eur": afr_median_pct_in_eur,
    "frac_afr_above_eur_75th": frac_afr_above_eur75,
    "frac_afr_below_eur_25th": frac_afr_below_eur25,
    "pct_afr_negative_bias": pct_afr_neg,
    "pct_eur_negative_bias": pct_eur_neg,
    "odds_ratio_negative_bias": float(or_neg),
    "chi2": float(chi2),
    "p_chi2": float(p_chi),
}
print(f"AFR median at {afr_median_pct_in_eur:.1f}th pct of EUR distribution")
print(f"AFR variants below EUR 25th pct: {100*frac_afr_below_eur25:.1f}%")
print(f"Negative bias: AFR={100*pct_afr_neg:.1f}% vs EUR={100*pct_eur_neg:.1f}%  OR={or_neg:.2f}x  p={p_chi:.3e}")

# ── 9. Bootstrap 95% CI on mean difference ─────────────────────────────────────
print("\n=== 9. Bootstrap 95% CI (NT-1000G EUR-AFR mean diff) ===")
rng = np.random.default_rng(42)
boot_diffs = []
for _ in range(10000):
    b_a = rng.choice(sc_a, size=len(sc_a), replace=True)
    b_e = rng.choice(sc_e, size=len(sc_e), replace=True)
    boot_diffs.append(float(np.mean(b_e) - np.mean(b_a)))
ci_lo, ci_hi = float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))
stats["bootstrap_ci_1000G"] = {"mean_diff": float(np.mean(boot_diffs)),
                                "ci95_lo": ci_lo, "ci95_hi": ci_hi,
                                "ci_all_positive": bool(ci_lo > 0)}
print(f"Mean diff (EUR-AFR): {np.mean(boot_diffs):+.4f}  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  all_positive={ci_lo>0}")

# ── 10. Save ──────────────────────────────────────────────────────────────────
with open(OUT, "w") as f:
    json.dump(stats, f, indent=2)
print(f"\nSaved -> {OUT}")
