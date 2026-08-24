# Training-Set Diversity Does Not Eliminate Ancestry-Associated Reference Bias in DNA Foundation Models

**Author:** Thackshanaramana B, Department of Computational Intelligence, SRM Institute of Science and Technology, Kattankulathur, Tamil Nadu, India

**Status:** Ready for arXiv submission → Briefings in Bioinformatics / Bioinformatics (Oxford)

---

## Summary

We show that Nucleotide Transformer (NT) exhibits a significant ancestry-associated scoring asymmetry: European-enriched common variants receive greater reference allele preference than African-enriched variants at matched within-population allele frequency (EUR−AFR gap Δ=+0.43, p<5×10⁻⁵, replicated across both NT models). Crucially, this gap changes by only **1.2%** when pretraining expands from a single reference genome (NT-hg38) to 3,202 diverse genomes (NT-1000G). Balanced resampling (1,000 iterations, N=509 vs N=509) confirms d=0.190 with the gap exceeding zero in **all 1,000 iterations** (95% significant). EUR-enriched variants are 1.32× over-represented in the top scoring decile.

---

## Repository structure

```
.
├── paper_D_ieee_final.tex       # Main paper (IEEEtran conference format)
├── references.bib               # BibTeX bibliography
├── figures/
│   ├── fig1_violin.png          # Bias score distributions by group and model
│   ├── fig2_calibration.png     # Within-group AF calibration scatter
│   └── fig3_ranking_displacement.png  # Cumulative ranking displacement analysis
├── scripts/
│   ├── fetch_gnomad_api.py      # Variant selection from gnomAD v4 GraphQL API
│   ├── variant_filter.py        # AFR/EUR enrichment filtering logic
│   ├── analyze_paper_d.py       # Primary statistical analysis (N=509 AF-matched)
│   ├── process_expanded_results.py  # Sensitivity analysis (N=6,830 EUR)
│   ├── resampling_analysis.py   # Balanced resampling + ranking displacement
│   ├── make_figures_paper_d.py  # Figure generation
│   └── utils.py                 # Shared utilities
├── results/
│   ├── paper_d_stats_2026-08-23.json             # Primary results (N=509)
│   ├── paper_d_stats_expanded_2026-08-23.json    # Sensitivity results (N=6,830)
│   └── paper_d_resampling_ranking_2026-08-24.json # Resampling + ranking results
└── kaggle_kernel/
    ├── output_final2/e1_full_chr22.jsonl    # Primary scored variants (N=509 EUR, AF-matched)
    └── output_expanded/e1_full_chr22.jsonl  # Sensitivity scored variants (N=6,830 EUR)
```

---

## Reproducing the analysis

**Requirements:** Python ≥ 3.9, `numpy`, `scipy`, `matplotlib`, `transformers`, `torch`

```bash
# 1. Primary statistical analysis
python scripts/analyze_paper_d.py

# 2. Balanced resampling + ranking displacement
python scripts/resampling_analysis.py

# 3. Generate figures
python scripts/make_figures_paper_d.py

# 4. Sensitivity analysis (expanded EUR)
python scripts/process_expanded_results.py
```

Scoring was performed on Kaggle (GPU T4 ×2) using `run_full_chr22.py`. The scored JSONL files are included in this repository (3.0 MB and 5.3 MB).

---

## Key results

| | NT-hg38 | NT-1000G |
|---|---|---|
| AFR mean bias score | +0.582 | +0.508 |
| EUR mean bias score | +1.010 | +0.941 |
| EUR−AFR gap (Δ) | +0.428 | +0.433 |
| p-value | 4.7×10⁻⁵ | 4.3×10⁻⁵ |
| Cohen's d (balanced) | 0.190 | 0.190 |
| Gap change hg38→1000G | — | **+1.2%** |

Balanced resampling (1,000×, N=509 vs 509): gap>0 in **100%** of iterations, p<0.05 in **94-95%** of iterations.

---

## Models

- `InstaDeepAI/nucleotide-transformer-500m-human-ref` (NT-hg38)
- `InstaDeepAI/nucleotide-transformer-500m-1000g` (NT-1000G)

## Data

- Variants: gnomAD v4 chromosome 22 SNVs
- African-enriched (AFR): AF_AFR≥10%, AF_EUR≤2%, fold≥5×; N=7,684
- European-enriched (EUR, primary): AF_EUR≥10%, AF_AFR≤2%; N=509 (AF-matched to AFR, Wilcoxon p=0.31)
- European-enriched (EUR, sensitivity): AF_EUR≥5%, AF_AFR≤2%; N=6,830

---

## Citation

```
Thackshanaramana B. "Training-Set Diversity Does Not Eliminate Ancestry-Associated
Reference Bias in DNA Foundation Models." 2026.
```
