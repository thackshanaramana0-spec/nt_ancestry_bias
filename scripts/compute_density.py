"""
Compute local variant density for AFR and EUR focal variants.
For each scored variant, count gnomAD common variants (AF_AFR>=0.05 OR AF_EUR>=0.05)
within +/-500 bp in the same chromosome.

Verifies numbers cited in paper §IV.E.
"""
import json, bisect
from pathlib import Path
import numpy as np
from scipy import stats as sp

BASE      = Path(__file__).parent.parent
GNOMAD    = BASE / "data/gnomad/gnomad_af_chr22.tsv"
JSONL     = BASE / "kaggle_kernel/output_final2/e1_full_chr22.jsonl"
RESULTS   = BASE / "results"
RESULTS.mkdir(exist_ok=True)

WINDOW = 500   # ±500 bp

# ── 1. Load all gnomAD common variants (AF_AFR>=0.05 or AF_EUR>=0.05) ─────────
print("Loading gnomAD TSV...")
common_pos = []   # sorted list of positions for binary-search window query
with open(GNOMAD) as f:
    header = f.readline().strip().split("\t")
    idx_pos = header.index("POS")
    idx_afr = header.index("AF_afr")
    idx_eur = header.index("AF_eur")
    for line in f:
        parts = line.strip().split("\t")
        try:
            pos     = int(parts[idx_pos])
            af_afr  = float(parts[idx_afr])
            af_eur  = float(parts[idx_eur])
        except (ValueError, IndexError):
            continue
        if af_afr >= 0.05 or af_eur >= 0.05:
            common_pos.append(pos)

common_pos.sort()
print(f"  Common variants (AF_AFR>=0.05 or AF_EUR>=0.05): {len(common_pos):,}")

def count_neighbours(focal_pos):
    """Count common variants within +/-WINDOW bp, excluding the focal position itself."""
    lo = bisect.bisect_left(common_pos, focal_pos - WINDOW)
    hi = bisect.bisect_right(common_pos, focal_pos + WINDOW)
    n = hi - lo
    # subtract focal variant itself if present
    if bisect.bisect_left(common_pos, focal_pos) < len(common_pos) and common_pos[bisect.bisect_left(common_pos, focal_pos)] == focal_pos:
        n -= 1
    return n

# ── 2. Load scored variants (unique positions; use NT-1000G AFR as representative) ─
print("Loading scored variant positions...")
afr_records = []   # (pos, bias_score)
eur_records = []

with open(JSONL) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("status") != "ok" or r.get("model") != "NT-1000G":
            continue
        if r["vtype"] == "ref_minor":
            afr_records.append((r["pos"], r["bias_score"]))
        elif r["vtype"] == "control":
            eur_records.append((r["pos"], r["bias_score"]))

print(f"  AFR focal variants: {len(afr_records)}")
print(f"  EUR focal variants: {len(eur_records)}")

# ── 3. Compute density for each focal variant ─────────────────────────────────
print("Computing densities (this takes ~30s)...")
afr_densities = [count_neighbours(pos) for pos, _ in afr_records]
afr_biases    = [bs for _, bs in afr_records]
eur_densities = [count_neighbours(pos) for pos, _ in eur_records]
eur_biases    = [bs for _, bs in eur_records]

afr_d = np.array(afr_densities)
eur_d = np.array(eur_densities)
afr_b = np.array(afr_biases)
eur_b = np.array(eur_biases)

print("\n=== DENSITY RESULTS ===")
print(f"AFR density: mean={np.mean(afr_d):.2f}  median={np.median(afr_d):.1f}  "
      f"IQR=[{np.percentile(afr_d,25):.0f},{np.percentile(afr_d,75):.0f}]")
print(f"EUR density: mean={np.mean(eur_d):.2f}  median={np.median(eur_d):.1f}  "
      f"IQR=[{np.percentile(eur_d,25):.0f},{np.percentile(eur_d,75):.0f}]")

t_dens, p_dens = sp.ttest_ind(afr_d, eur_d, equal_var=False)
print(f"Density Welch t={t_dens:.3f}  p={p_dens:.3e}")

# ── 4. Within-group density-bias correlation ───────────────────────────────────
r_afr, p_afr = sp.pearsonr(afr_d, afr_b)
r_eur, p_eur = sp.pearsonr(eur_d, eur_b)
print(f"\nWithin-group density-bias correlation:")
print(f"  AFR: r={r_afr:+.4f}  p={p_afr:.3e}")
print(f"  EUR: r={r_eur:+.4f}  p={p_eur:.3e}")

# ── 5. Density-matched AFR subset ─────────────────────────────────────────────
eur_iqr_lo = float(np.percentile(eur_d, 25))
eur_iqr_hi = float(np.percentile(eur_d, 75))
mask = (afr_d >= eur_iqr_lo) & (afr_d <= eur_iqr_hi)
afr_matched_b = afr_b[mask]
print(f"\nDensity-matched AFR (IQR {eur_iqr_lo:.0f}–{eur_iqr_hi:.0f}): "
      f"N={np.sum(mask)}, mean bias={np.mean(afr_matched_b):+.3f}")
t_m, p_m = sp.ttest_ind(afr_matched_b, eur_b, equal_var=False)
gap_matched = float(np.mean(eur_b) - np.mean(afr_matched_b))
print(f"  EUR mean={np.mean(eur_b):+.3f}, gap={gap_matched:+.3f}, p={p_m:.3e}")

# ── 6. Save ───────────────────────────────────────────────────────────────────
density_stats = {
    "AFR": {"mean": float(np.mean(afr_d)), "median": float(np.median(afr_d)),
            "iqr_lo": float(np.percentile(afr_d, 25)), "iqr_hi": float(np.percentile(afr_d, 75))},
    "EUR": {"mean": float(np.mean(eur_d)), "median": float(np.median(eur_d)),
            "iqr_lo": float(np.percentile(eur_d, 25)), "iqr_hi": float(np.percentile(eur_d, 75))},
    "density_welch_t": float(t_dens), "density_p": float(p_dens),
    "density_bias_corr_AFR": {"r": float(r_afr), "p": float(p_afr)},
    "density_bias_corr_EUR": {"r": float(r_eur), "p": float(p_eur)},
    "density_matched_AFR": {
        "eur_iqr_range": [eur_iqr_lo, eur_iqr_hi],
        "n": int(np.sum(mask)),
        "mean_bias": float(np.mean(afr_matched_b)),
        "gap_with_EUR": gap_matched,
        "p": float(p_m),
    }
}

import json as _json
out = RESULTS / "paper_d_density_2026-08-25.json"
with open(out, "w") as f:
    _json.dump(density_stats, f, indent=2)
print(f"\nSaved -> {out}")

print("\n=== NUMBERS FOR PAPER (verify §IV.E) ===")
print(f"AFR mean density: {np.mean(afr_d):.2f}  EUR mean density: {np.mean(eur_d):.2f}")
print(f"Density comparison p={p_dens:.2e}")
print(f"AFR density-bias r={r_afr:+.4f} p={p_afr:.3e}")
print(f"EUR IQR density: [{eur_iqr_lo:.0f},{eur_iqr_hi:.0f}]")
print(f"Density-matched AFR bias: {np.mean(afr_matched_b):+.3f}  gap with EUR: {gap_matched:+.3f}  p={p_m:.3e}")
