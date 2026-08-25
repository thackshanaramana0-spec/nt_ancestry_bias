"""
Experiment 2: Strict nearest-neighbour within-population AF matching.

For each EUR variant (AF_EUR = x), match to the AFR variant with closest
AF_AFR = x, without replacement.  Then rerun all primary statistics on
matched N=509 vs N=509 pairs.

Reads: kaggle_kernel/output_final2/e1_full_chr22.jsonl
Saves: results/paper_d_af_matched_2026-08-25.json
"""

import json
import bisect
from pathlib import Path

import numpy as np
from scipy import stats as sp

BASE  = Path(__file__).parent.parent
JSONL = BASE / "kaggle_kernel/output_final2/e1_full_chr22.jsonl"
OUT   = BASE / "results/paper_d_af_matched_2026-08-25.json"
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

def get(model, vtype):
    return [r for r in records if r["model"] == model and r["vtype"] == vtype]

results = {}

for model in ["NT-hg38", "NT-1000G"]:
    afr = get(model, "ref_minor")
    eur = get(model, "control")

    # Sort AFR pool by AF_AFR for fast nearest-neighbour search
    afr_sorted = sorted(afr, key=lambda r: r["af_afr"])
    afr_af_sorted = [r["af_afr"] for r in afr_sorted]
    used = [False] * len(afr_sorted)

    matched_afr = []
    matched_eur = list(eur)   # all 509 EUR variants

    for e in matched_eur:
        target = e["af_eur"]
        # Binary search for insertion point
        idx = bisect.bisect_left(afr_af_sorted, target)
        # Check neighbours (left and right) for unused minimum
        best_idx, best_dist = None, float("inf")
        for candidate in range(max(0, idx - 200), min(len(afr_sorted), idx + 201)):
            if not used[candidate]:
                dist = abs(afr_af_sorted[candidate] - target)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = candidate
        if best_idx is None:
            raise RuntimeError("No unused AFR match found — AFR pool exhausted")
        used[best_idx] = True
        matched_afr.append(afr_sorted[best_idx])

    sc_a = np.array([r["bias_score"] for r in matched_afr])
    sc_e = np.array([r["bias_score"] for r in matched_eur])
    af_a = np.array([r["af_afr"]     for r in matched_afr])
    af_e = np.array([r["af_eur"]     for r in matched_eur])

    t, p = sp.ttest_ind(sc_e, sc_a, equal_var=False)
    diff = float(np.mean(sc_e) - np.mean(sc_a))
    na, ne = len(sc_a), len(sc_e)
    pooled_sd = float(np.sqrt(
        ((na-1)*np.var(sc_a, ddof=1) + (ne-1)*np.var(sc_e, ddof=1)) / (na+ne-2)
    ))
    d = diff / pooled_sd

    U, p_mw = sp.mannwhitneyu(sc_e, sc_a, alternative="greater")
    auroc = float(U) / (ne * na)

    rng = np.random.default_rng(42)
    boot = [float(np.mean(rng.choice(sc_e, ne, replace=True)) -
                  np.mean(rng.choice(sc_a, na, replace=True)))
            for _ in range(10_000)]
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # AF matching quality
    af_match_quality = float(np.mean(np.abs(af_a - af_e)))

    key = model.replace("-", "_")
    results[key] = {
        "n_eur": ne, "n_afr_matched": na,
        "af_match_mean_abs_diff": af_match_quality,
        "afr_af_mean": float(np.mean(af_a)), "eur_af_mean": float(np.mean(af_e)),
        "eur_mean_score": float(np.mean(sc_e)), "afr_mean_score": float(np.mean(sc_a)),
        "diff_eur_minus_afr": diff,
        "t": float(t), "p": float(p),
        "cohens_d": d,
        "auroc": auroc, "p_mannwhitney": float(p_mw),
        "ci95_lo": ci_lo, "ci95_hi": ci_hi,
        "ci_all_positive": bool(ci_lo > 0),
    }
    print(f"\n{model} — AF-matched N=509 vs N=509")
    print(f"  AF match quality: mean |AF_AFR - AF_EUR| = {af_match_quality:.4f}")
    print(f"  AFR AF mean: {np.mean(af_a):.4f}   EUR AF mean: {np.mean(af_e):.4f}")
    print(f"  EUR mean score: {np.mean(sc_e):+.4f}   AFR mean score: {np.mean(sc_a):+.4f}")
    print(f"  gap(EUR-AFR) = {diff:+.4f}  t={t:.3f}  p={p:.3e}  d={d:.4f}  AUROC={auroc:.4f}")
    print(f"  Bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]  all_positive={ci_lo>0}")

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUT}")
