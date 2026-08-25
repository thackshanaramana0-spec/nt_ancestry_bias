"""
Experiment 3: Reference-context divergence as mechanistic test.

Proxy for local haplotype divergence from GRCh38:
  afr_context_score = count of AFR-enriched gnomAD variants within +-500 bp
    (AF_AFR >= 0.10, AF_EUR <= 0.02, fold >= 5x)
  This captures how many nearby positions have African-specific alleles,
  approximating how much the local haplotype context diverges from the
  European-anchored reference in African-ancestry haplotypes.

Tests:
  1. Correlation of afr_context_score with bias_score within AFR group
  2. OLS: bias_score ~ group_label + afr_context_score + within_pop_AF
     -- if group_label beta remains significant, ancestry gap is not explained
     by reference-context divergence alone

Reads: gnomad_af_chr22.tsv + e1_full_chr22.jsonl
Saves: results/paper_d_haplotype_divergence_2026-08-25.json
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats as sp

BASE   = Path(__file__).parent.parent
GNOMAD = BASE / "data/gnomad/gnomad_af_chr22.tsv"
JSONL  = BASE / "kaggle_kernel/output_final2/e1_full_chr22.jsonl"
OUT    = BASE / "results/paper_d_haplotype_divergence_2026-08-25.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

WINDOW = 500   # bp each side

# ── Load gnomAD variants ──────────────────────────────────────────────────────
print("Loading gnomAD chr22 variants...")
gnomad_pos = []
gnomad_af_afr = []
gnomad_af_eur = []

with open(GNOMAD) as f:
    header = f.readline().strip().split("\t")
    pos_i   = header.index("POS")
    afr_i   = header.index("AF_afr")
    eur_i   = header.index("AF_eur")
    for line in f:
        parts = line.strip().split("\t")
        try:
            pos  = int(parts[pos_i])
            afr  = float(parts[afr_i])
            eur  = float(parts[eur_i])
        except (ValueError, IndexError):
            continue
        gnomad_pos.append(pos)
        gnomad_af_afr.append(afr)
        gnomad_af_eur.append(eur)

gnomad_pos    = np.array(gnomad_pos,    dtype=np.int64)
gnomad_af_afr = np.array(gnomad_af_afr, dtype=np.float32)
gnomad_af_eur = np.array(gnomad_af_eur, dtype=np.float32)

# Pre-compute AFR-enriched indicator (same criteria as study)
afr_enriched_mask = (
    (gnomad_af_afr >= 0.10) &
    (gnomad_af_eur <= 0.02) &
    ((gnomad_af_afr / np.where(gnomad_af_eur > 0, gnomad_af_eur, 1e-6)) >= 5.0)
)
print(f"gnomAD chr22 loaded: {len(gnomad_pos):,} variants, "
      f"{afr_enriched_mask.sum():,} AFR-enriched")

sort_idx = np.argsort(gnomad_pos)
gnomad_pos_sorted = gnomad_pos[sort_idx]
afr_enriched_sorted = afr_enriched_mask[sort_idx]

def afr_context_count(focal_pos: int) -> int:
    lo = np.searchsorted(gnomad_pos_sorted, focal_pos - WINDOW, side="left")
    hi = np.searchsorted(gnomad_pos_sorted, focal_pos + WINDOW, side="right")
    window_slice = afr_enriched_sorted[lo:hi]
    # exclude the focal variant itself if present
    focal_in_window = gnomad_pos_sorted[lo:hi] == focal_pos
    return int(window_slice.sum()) - int((window_slice & focal_in_window).sum())

# ── Load scored variants ───────────────────────────────────────────────────────
print("Loading scored variants...")
records = []
with open(JSONL) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("status") == "ok" and r.get("bias_score") is not None:
            records.append(r)

def get(model, vtype):
    return [r for r in records if r["model"] == model and r["vtype"] == vtype]

results = {}

for model in ["NT-hg38", "NT-1000G"]:
    print(f"\nComputing AFR-context divergence scores for {model}...")
    afr_recs = get(model, "ref_minor")
    eur_recs  = get(model, "control")
    all_recs  = afr_recs + eur_recs

    # Compute AFR-context score for each variant
    for r in all_recs:
        r["_afr_ctx"] = afr_context_count(int(r["pos"]))

    afr_ctx_a = np.array([r["_afr_ctx"] for r in afr_recs], dtype=float)
    afr_ctx_e = np.array([r["_afr_ctx"] for r in eur_recs],  dtype=float)
    sc_a      = np.array([r["bias_score"] for r in afr_recs])
    sc_e      = np.array([r["bias_score"] for r in eur_recs])
    af_a      = np.array([r["af_afr"]     for r in afr_recs])
    af_e      = np.array([r["af_eur"]     for r in eur_recs])

    # 1. AFR-context density between groups
    t_ctx, p_ctx = sp.ttest_ind(afr_ctx_a, afr_ctx_e, equal_var=False)
    print(f"  AFR-context density: AFR={np.mean(afr_ctx_a):.2f}  EUR={np.mean(afr_ctx_e):.2f}"
          f"  t={t_ctx:.2f}  p={p_ctx:.3e}")

    # 2. Correlation of afr_context with bias_score within AFR group
    r_ctx_bias, p_ctx_bias = sp.pearsonr(afr_ctx_a, sc_a)
    print(f"  AFR group: r(afr_context, bias) = {r_ctx_bias:+.4f}  p={p_ctx_bias:.3e}")

    # 3. OLS: bias ~ group + afr_context + within_pop_AF
    #    group_label: 0=AFR, 1=EUR
    n_afr, n_eur = len(sc_a), len(sc_e)
    group_all    = np.array([0]*n_afr + [1]*n_eur, dtype=float)
    ctx_all      = np.concatenate([afr_ctx_a, afr_ctx_e])
    af_all       = np.concatenate([af_a, af_e])
    score_all    = np.concatenate([sc_a, sc_e])

    # Design matrix: intercept, group, afr_context, within_pop_AF
    X = np.column_stack([
        np.ones(n_afr + n_eur),
        group_all,
        ctx_all,
        af_all,
    ])
    # OLS via least squares
    beta, _, _, _ = np.linalg.lstsq(X, score_all, rcond=None)
    resid = score_all - X @ beta
    n, k = X.shape
    mse   = float(np.sum(resid**2) / (n - k))
    cov   = mse * np.linalg.inv(X.T @ X)
    se    = np.sqrt(np.diag(cov))
    t_reg = beta / se
    p_reg = 2 * sp.t.sf(np.abs(t_reg), df=n-k)

    print(f"  OLS: beta_group={beta[1]:+.4f} (SE={se[1]:.4f}) t={t_reg[1]:.3f} p={p_reg[1]:.3e}")
    print(f"       beta_afr_ctx={beta[2]:+.4f} t={t_reg[2]:.3f} p={p_reg[2]:.3e}")
    print(f"       beta_af={beta[3]:+.4f} t={t_reg[3]:.3f} p={p_reg[3]:.3e}")

    key = model.replace("-", "_")
    results[key] = {
        "afr_ctx_mean_afr_group":  float(np.mean(afr_ctx_a)),
        "afr_ctx_mean_eur_group":  float(np.mean(afr_ctx_e)),
        "t_ctx_between_groups":    float(t_ctx),
        "p_ctx_between_groups":    float(p_ctx),
        "pearson_r_ctx_bias_AFR":  float(r_ctx_bias),
        "p_r_ctx_bias_AFR":        float(p_ctx_bias),
        "ols_beta_group":          float(beta[1]),
        "ols_se_group":            float(se[1]),
        "ols_t_group":             float(t_reg[1]),
        "ols_p_group":             float(p_reg[1]),
        "ols_beta_afr_ctx":        float(beta[2]),
        "ols_t_afr_ctx":           float(t_reg[2]),
        "ols_p_afr_ctx":           float(p_reg[2]),
        "group_significant_after_controlling_divergence": bool(p_reg[1] < 0.05),
    }

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUT}")
