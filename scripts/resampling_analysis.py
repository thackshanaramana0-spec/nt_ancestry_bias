"""
Balanced resampling + ranking displacement analysis for Paper D.
Addresses two reviewer concerns:
  1. Sample imbalance (7684 AFR vs 509 EUR) - balanced resampling fixes this
  2. Small Cohen's d - ranking displacement shows cumulative population-level effect
"""

import json, random, math, statistics
import math as _math
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent.parent
JSONL_ORIG = BASE / "kaggle_kernel" / "output_final2" / "e1_full_chr22.jsonl"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── load original AF-matched data ────────────────────────────────────────────
print("Loading original AF-matched data (N_EUR=509)...")
records = []
with open(JSONL_ORIG) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

print(f"  Total records loaded: {len(records)}")

# index by model × group
by_model_group = defaultdict(lambda: defaultdict(list))
for r in records:
    if r.get("status") != "ok":
        continue
    model = r.get("model", "")
    group = r.get("vtype", "")   # field is vtype: "ref_minor" or "control"
    bias = r.get("bias_score")
    af_afr = r.get("af_afr", None)
    af_eur = r.get("af_eur", None)
    if bias is None:
        continue
    by_model_group[model][group].append({
        "bias": bias,
        "af_afr": af_afr,
        "af_eur": af_eur,
    })

models = list(by_model_group.keys())
print(f"  Models: {models}")
for m in models:
    for g, recs in by_model_group[m].items():
        print(f"    {m} / {g}: N={len(recs)}")

# ── helper: mean, std, cohen's d, t-stat ─────────────────────────────────────
def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va = statistics.variance(a)
    vb = statistics.variance(b)
    pooled_sd = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_sd == 0:
        return float("nan")
    return (statistics.mean(a) - statistics.mean(b)) / pooled_sd

def welch_t(a, b):
    na, nb = len(a), len(b)
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, 1.0
    t = (ma - mb) / se
    z = abs(t)
    p = 2 * (1 - _norm_cdf(z))
    return t, p

def _norm_cdf(x):
    return 0.5 * (1 + _math.erf(x / _math.sqrt(2)))

# ── 1. BALANCED RESAMPLING ────────────────────────────────────────────────────
print("\n" + "="*60)
print("BALANCED RESAMPLING (1000 iterations, N=509 vs N=509)")
print("="*60)

N_ITER = 1000
random.seed(42)

resampling_results = {}

for model in models:
    afr_recs = by_model_group[model].get("ref_minor", [])
    eur_recs = by_model_group[model].get("control", [])

    if not afr_recs or not eur_recs:
        print(f"  {model}: missing groups, skip")
        continue

    n_eur = len(eur_recs)
    n_afr = len(afr_recs)
    n_sample = min(n_eur, n_afr, 509)  # 509 = original EUR size
    print(f"\n  {model}: AFR pool={n_afr}, EUR={n_eur}, sample_size={n_sample}")

    eur_biases = [r["bias"] for r in eur_recs]
    afr_biases = [r["bias"] for r in afr_recs]

    gaps = []
    ds = []
    significant = 0

    for _ in range(N_ITER):
        afr_sample = random.sample(afr_biases, n_sample)
        eur_sample = random.sample(eur_biases, min(n_sample, n_eur))
        gap = statistics.mean(eur_sample) - statistics.mean(afr_sample)
        d = cohens_d(eur_sample, afr_sample)
        _, p = welch_t(eur_sample, afr_sample)
        gaps.append(gap)
        ds.append(d)
        if p < 0.05:
            significant += 1

    median_gap = statistics.median(gaps)
    median_d = statistics.median(ds)
    pct_sig = 100 * significant / N_ITER
    pct_positive = 100 * sum(1 for g in gaps if g > 0) / N_ITER

    # 95% CI of gap distribution
    gaps_sorted = sorted(gaps)
    ci_lo = gaps_sorted[int(0.025 * N_ITER)]
    ci_hi = gaps_sorted[int(0.975 * N_ITER)]

    print(f"    Median gap (EUR-AFR): {median_gap:+.4f}")
    print(f"    95% CI of gap:        [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"    Median Cohen's d:     {median_d:.4f}")
    print(f"    % iterations gap>0:  {pct_positive:.1f}%")
    print(f"    % iterations p<0.05: {pct_sig:.1f}%")

    resampling_results[model] = {
        "n_sample": n_sample,
        "n_iterations": N_ITER,
        "median_gap": median_gap,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "median_cohens_d": median_d,
        "pct_gap_positive": pct_positive,
        "pct_significant": pct_sig,
    }

# ── 2. RANKING DISPLACEMENT ───────────────────────────────────────────────────
print("\n" + "="*60)
print("RANKING DISPLACEMENT ANALYSIS")
print("="*60)

ranking_results = {}

for model in models:
    afr_recs = by_model_group[model].get("ref_minor", [])
    eur_recs = by_model_group[model].get("control", [])
    if not afr_recs or not eur_recs:
        continue

    # pool all variants and rank by bias score (highest = most favored)
    all_biases = [(r["bias"], "AFR") for r in afr_recs] + \
                 [(r["bias"], "EUR") for r in eur_recs]
    all_biases.sort(key=lambda x: x[0], reverse=True)  # descending

    N_total = len(all_biases)
    afr_ranks = []
    eur_ranks = []
    for rank, (bias, group) in enumerate(all_biases, 1):
        if group == "AFR":
            afr_ranks.append(rank)
        else:
            eur_ranks.append(rank)

    # percentile = where in the ranked list (lower rank = better = higher bias)
    # percentile score: (N - rank) / (N - 1) * 100  → 100 = best
    afr_pct = [(N_total - r) / (N_total - 1) * 100 for r in afr_ranks]
    eur_pct = [(N_total - r) / (N_total - 1) * 100 for r in eur_ranks]

    mean_afr_pct = statistics.mean(afr_pct)
    mean_eur_pct = statistics.mean(eur_pct)

    # top/bottom decile composition
    top10_n = N_total // 10
    bot10_n = N_total // 10
    top10 = all_biases[:top10_n]
    bot10 = all_biases[-bot10_n:]

    top10_afr = sum(1 for _, g in top10 if g == "AFR")
    top10_eur = sum(1 for _, g in top10 if g == "EUR")
    bot10_afr = sum(1 for _, g in bot10 if g == "AFR")
    bot10_eur = sum(1 for _, g in bot10 if g == "EUR")

    n_afr = len(afr_recs)
    n_eur = len(eur_recs)

    # expected proportions (by size)
    exp_afr_top10 = n_afr / N_total * top10_n
    exp_eur_top10 = n_eur / N_total * top10_n

    # fraction displaced: AFR variants ranking in bottom half
    afr_bottom_half = sum(1 for r in afr_ranks if r > N_total / 2)
    afr_bottom_half_pct = 100 * afr_bottom_half / n_afr

    print(f"\n  {model}:")
    print(f"    Total variants pooled: {N_total} (AFR={n_afr}, EUR={n_eur})")
    print(f"    Mean percentile rank — AFR: {mean_afr_pct:.1f}th, EUR: {mean_eur_pct:.1f}th")
    print(f"    Top-10% of variants ({top10_n} total):")
    print(f"      AFR: {top10_afr} observed vs {exp_afr_top10:.0f} expected ({100*top10_afr/top10_n:.1f}% of top decile)")
    print(f"      EUR: {top10_eur} observed vs {exp_eur_top10:.0f} expected ({100*top10_eur/top10_n:.1f}% of top decile)")
    print(f"    Bottom-10% of variants ({bot10_n} total):")
    print(f"      AFR: {bot10_afr} ({100*bot10_afr/bot10_n:.1f}% of bottom decile)")
    print(f"      EUR: {bot10_eur} ({100*bot10_eur/bot10_n:.1f}% of bottom decile)")
    print(f"    AFR variants in bottom half of ranking: {afr_bottom_half_pct:.1f}%")

    # cumulative: how many AFR variants are in top-K% vs expected
    thresholds = [10, 20, 25, 33, 50]
    print(f"    Cumulative representation (AFR observed vs expected in top-K%):")
    cumulative = {}
    for pct_thresh in thresholds:
        k = int(N_total * pct_thresh / 100)
        obs = sum(1 for _, g in all_biases[:k] if g == "AFR")
        exp = n_afr * pct_thresh / 100
        ratio = obs / exp if exp > 0 else float("nan")
        print(f"      top-{pct_thresh}%: AFR obs={obs}, exp={exp:.0f}, ratio={ratio:.3f}")
        cumulative[f"top{pct_thresh}pct"] = {"obs": obs, "exp": exp, "ratio": ratio}

    ranking_results[model] = {
        "N_total": N_total,
        "n_afr": n_afr,
        "n_eur": n_eur,
        "mean_afr_percentile": mean_afr_pct,
        "mean_eur_percentile": mean_eur_pct,
        "top10_afr_obs": top10_afr,
        "top10_afr_exp": exp_afr_top10,
        "top10_eur_obs": top10_eur,
        "top10_eur_exp": exp_eur_top10,
        "afr_bottom_half_pct": afr_bottom_half_pct,
        "cumulative": cumulative,
    }

# ── 3. SAVE ───────────────────────────────────────────────────────────────────
output = {
    "balanced_resampling": resampling_results,
    "ranking_displacement": ranking_results,
}

out_path = RESULTS_DIR / "paper_d_resampling_ranking_2026-08-24.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {out_path}")
print("\n=== KEY NUMBERS FOR PAPER ===")
for model in models:
    if model in resampling_results:
        r = resampling_results[model]
        rk = ranking_results.get(model, {})
        print(f"\n{model}:")
        print(f"  Balanced resampling: gap={r['median_gap']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}], "
              f"d={r['median_cohens_d']:.4f}, {r['pct_gap_positive']:.0f}% positive, "
              f"{r['pct_significant']:.0f}% significant")
        if rk:
            print(f"  AFR mean percentile: {rk['mean_afr_percentile']:.1f}th vs EUR {rk['mean_eur_percentile']:.1f}th")
            print(f"  AFR in bottom half of ranking: {rk['afr_bottom_half_pct']:.1f}%")
